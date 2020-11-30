#!/usr/bin/env python3
from argparse import ArgumentParser
import json
import numpy as np
import pandas as pd
import pickle
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import tqdm
from xgboost import XGBClassifier

import cv_search
import extract_features as ef
from evaluation_utils import quick_evaluate
from nn import PatchedNeuralNetClassifier
import xgb_wrapped
from sample_utils import (
    sample_imbalanced, )


def adjust_train_data(model, y, ixs=None, seed=None):
    if ixs is None:
        ixs = np.arange(0, y.shape[0])

    if isinstance(model, (
            LogisticRegression,
            PatchedNeuralNetClassifier,
    )):
        return ixs
    elif isinstance(model, RandomForestClassifier):
        return ixs
    elif isinstance(model, XGBClassifier):
        # sample to avoid OOM
        return sample_imbalanced(y, ixs, 0.2, seed=seed)
    else:
        print("No modification specified for", type(model))
        return ixs


def adjust_model(model, y):
    if isinstance(model, XGBClassifier):
        scale_pos_weight = cv_search.get_scale_pos_weight(y)
        model.set_params(scale_pos_weight=scale_pos_weight)
    return model


def init_nn(model, diags):
    return model.set_params(module__input_size=len(diags))


def set_model_hyperparams(model, model_name, cv_param_summary):
    if cv_param_summary is None:
        print("No hyperparameter info")
        return model
    if isinstance(cv_param_summary, pd.DataFrame):
        params = cv_param_summary[cv_param_summary["model"] == model_name]
        if params.shape[0] == 0:
            print("No hyperparameters for", model_name)
            return model
        params = params.sort_values("score", ascending=False)
        chosen_config = params.iloc[0]["params"]
    else:
        chosen_config = cv_param_summary[model_name]
    print(model_name, "running with", chosen_config)
    model.set_params(**chosen_config)
    return model


def get_cv_iter(X, y, n_splits, model=None, seed=None):
    splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    split = splitter.split(X, y)
    for iter_ix, (train_ix, test_ix) in enumerate(split):
        # only adjust training data, not test
        if model is not None:
            train_ix = adjust_train_data(
                model,
                y,
                train_ix,
                seed=seed + iter_ix if seed is not None else None,
            )
        yield train_ix, test_ix


def cv_train_test(model, X, y, cv_iter, bootstrap_iters=1000):
    results = []
    for ix, (train_ix, test_ix) in tqdm.tqdm(enumerate(cv_iter)):
        X_train, y_train = X[train_ix], y[train_ix]
        X_test, y_test = X[test_ix], y[test_ix]
        model.fit(X_train, y_train)
        y_probs = model.predict_proba(X_test)
        res_iter = quick_evaluate(
            y_test,
            y_probs,
            bootstrap_iters=bootstrap_iters,
        )
        res_iter["ix"] = ix
        res_iter["train_ix"] = train_ix
        res_iter["test_ix"] = test_ix
        results.append(res_iter)
    results_df = pd.DataFrame(results)
    return results_df


def single_fit(model, X_train, y_train, seed):
    adj_ix = adjust_train_data(model, y_train, seed=seed)
    X_train, y_train = X_train[adj_ix], y_train[adj_ix]

    model = adjust_model(model, y_train)
    model.fit(X_train, y_train)
    return model


def just_train(model, bidmc, seed):
    X_train, y_train = bidmc
    return single_fit(model, X_train, y_train, seed)


def external_performance(
        model,
        bidmc,
        phc,
        n_splits,
        seed,
        bootstrap_iters=1000,
        prefit=False,
        test_uids=None,
):
    # case where we want to use all test data
    # no CV splitting
    if n_splits == 0:
        results = test_performance(
            model,
            bidmc,
            phc,
            seed,
            bootstrap_iters=bootstrap_iters,
            prefit=prefit,
            test_uids=test_uids,
        )
        return results

    # fit once, with BIDMC data
    if not prefit:
        X_train, y_train = bidmc
        model = single_fit(model, X_train, y_train, seed=seed)

    # calibrate and evaluate using CV on PHC
    X_test, y_test = phc
    model = CalibratedClassifierCV(
        model,
        method="sigmoid",
        cv="prefit",
    )
    cv_iter = get_cv_iter(X_test, y_test, n_splits, seed=seed)
    return cv_train_test(
        model,
        X_test,
        y_test,
        cv_iter,
        bootstrap_iters=bootstrap_iters,
    )


def test_performance(
        model,
        bidmc_train,
        bidmc_test,
        seed,
        bootstrap_iters=1000,
        prefit=False,
        test_uids=None,
):
    # fit once, with BIDMC training data
    if not prefit:
        print("Refitting")
        X_train, y_train = bidmc_train
        model = single_fit(model, X_train, y_train, seed)
    # evaluate on test split
    X_test, y_test = bidmc_test
    y_probs = model.predict_proba(X_test)
    results = quick_evaluate(
        y_test,
        y_probs,
        bootstrap_iters=bootstrap_iters,
        uids=test_uids,
    )
    return pd.DataFrame([results])


def train_performance(
        model,
        bidmc_train,
        n_splits,
        seed,
        bootstrap_iters=1000,
):
    # evaluate using CV on training data
    X_train, y_train = bidmc_train
    model = adjust_model(model, y_train)

    cv_iter = get_cv_iter(X_train, y_train, n_splits, model=model, seed=seed)
    return cv_train_test(
        model,
        X_train,
        y_train,
        cv_iter,
        bootstrap_iters=bootstrap_iters,
    )


def get_trained_model(model_name, cutoff, trained_models):
    """
    Retrieve trained model from list of (model_name, cutoff, model)
    entries.
    """
    d = {(m, c): tm for m, c, tm in trained_models}
    return d[(model_name, cutoff)]


def get_and_init_untrained_model(model_name, feats, cv_param_summary):
    # otherwise we'll initialize model and set
    # hyperparameters
    if model_name == "xgb-ensemble":
        model = cv_search.get_model("xgb")
    else:
        model = cv_search.get_model(model_name)

    if isinstance(model, PatchedNeuralNetClassifier):
        model = init_nn(model, feats)
    # we only set hyperparameters
    # when not loading from disk
    model = set_model_hyperparams(
        model,
        "xgb" if model_name == "xgb-ensemble" else model_name,
        cv_param_summary,
    )
    if model_name == "xgb-ensemble":
        model = xgb_wrapped.xgb_ensemble(
            model,
            max_samples=10000,
            sampling_strategy="auto",
            n_jobs=1,
        )
    return model


def add_demographics(demo_df, target_df, X_and_y, cutoff):
    if X_and_y is None:
        return None
    X, y = X_and_y
    demo_df["UID"] = demo_df["UID"].astype(str)
    target_df["UID"] = target_df["UID"].astype(str)
    demo_df = demo_df[demo_df.UID.isin(target_df.UID.values)]
    demo_df = demo_df.sort_values("UID", ascending=False)
    age_vec = demo_df["orig_age"].fillna(0.0).values
    age_buckets_df = ef.create_age_buckets(age_vec, cutoff)
    # drop anything we haven't explicitly encoded ourselves
    orig_demo_cols = [c for c in demo_df.columns if c.startswith("orig_")]
    demo_df = demo_df.drop(columns=["UID"] + orig_demo_cols)
    demo_X = np.hstack((demo_df.values, age_buckets_df.values))
    assert X.shape[0] == demo_X.shape[0], "Missing rows"
    X = np.hstack((X, demo_X))
    return X, y


def get_args():
    parser = ArgumentParser(description="Carry out model validation")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "train-performance",
            "test-performance",
            "external-performance",
            "just-train",
        ],
        help="Version of validation",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Names for models to evaluate (from cv_search)",
    )
    parser.add_argument("--train", type=str, help="Training data")
    parser.add_argument("--test", type=str, help="Test data")
    parser.add_argument(
        "--hyperparameters",
        type=str,
        help=
        "Path to pickled hyperparameters summary dataframe (from cv_search)",
    )
    parser.add_argument("--specific_hyperparameters",
                        type=str,
                        help="Dictionary of hyperparameters to use")
    parser.add_argument(
        "--features",
        type=str,
        help="Path to pickled list of diagnoses or feature map",
    )
    parser.add_argument(
        "--trained_models",
        type=str,
        help=
        "Path to pickled list of (model_name, cutoff, model) with trained models",
    )
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        help="Prediction cutoffs (censoring)",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        help="Number of splits if performing CV",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for pickled results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="RNG seed",
    )
    parser.add_argument(
        "--bootstrap_iters",
        type=int,
        help="Bootstrap iterations",
        default=1000,
    )
    parser.add_argument(
        "--train_demographics",
        type=str,
        help="Path to demographics data for train data",
    )
    parser.add_argument(
        "--test_demographics",
        type=str,
        help="Path to demographics data for test data",
    )
    return parser.parse_args()


def main():
    args = get_args()
    train_df, train_xy = None, None
    if args.train:
        train_df = pd.read_feather(args.train)

    test_df, test_xy = None, None
    if args.test:
        test_df = pd.read_feather(args.test)

    feats = pd.read_pickle(args.features)

    if args.hyperparameters:
        cv_param_summary = pd.read_pickle(args.hyperparameters)
    elif args.specific_hyperparameters:
        cv_param_summary = json.loads(args.specific_hyperparameters)
    else:
        cv_param_summary = None

    trained_models = None
    if args.trained_models:
        with open(args.trained_models, "rb") as fin:
            trained_models = pickle.load(fin)

    train_demographics_df = None
    if args.train_demographics:
        train_demographics_df = pd.read_feather(args.train_demographics)

    test_demographics_df = None
    if args.test_demographics:
        test_demographics_df = pd.read_feather(args.test_demographics)

    results = []
    for cutoff in args.cutoffs:
        if train_df is not None:
            pruned_train_df = train_df[train_df.date >= cutoff]
            train_xy = ef.get_matrix_and_y(
                pruned_train_df,
                cutoff,
                np.inf,
                feats,
            )

        if test_df is not None:
            pruned_test_df = test_df[test_df.date >= cutoff]
            test_xy, test_uids = ef.get_matrix_and_y(
                pruned_test_df,
                cutoff,
                np.inf,
                feats,
                return_uids=True,
            )

        # add demographic data if provided
        if train_demographics_df is not None:
            train_xy = add_demographics(
                train_demographics_df,
                pruned_train_df,
                train_xy,
                cutoff,
            )

        if test_demographics_df is not None:
            test_xy = add_demographics(
                test_demographics_df,
                pruned_test_df,
                test_xy,
                cutoff,
            )

        for model_name in tqdm.tqdm(args.models):
            if trained_models is not None:
                print("Using pre-trained model", model_name, "for", cutoff)
                model = get_trained_model(model_name, cutoff, trained_models)
            else:
                model = get_and_init_untrained_model(
                    model_name,
                    feats,
                    cv_param_summary,
                )
            if args.mode == "train-performance":
                res = train_performance(
                    model,
                    train_xy,
                    args.n_splits,
                    args.seed,
                    bootstrap_iters=args.bootstrap_iters,
                )
            elif args.mode == "test-performance":
                res = test_performance(
                    model,
                    train_xy,
                    test_xy,
                    args.seed,
                    bootstrap_iters=args.bootstrap_iters,
                    prefit=trained_models is not None,
                    test_uids=test_uids,
                )
            elif args.mode == "external-performance":
                res = external_performance(
                    model,
                    train_xy,
                    test_xy,
                    args.n_splits,
                    args.seed,
                    bootstrap_iters=args.bootstrap_iters,
                    prefit=trained_models is not None,
                    test_uids=test_uids,
                )
            elif args.mode == "just-train":
                trained_model = just_train(model, train_xy, args.seed)
                res = (model_name, cutoff, trained_model)
            else:
                raise ValueError("Unknown mode:", args.mode)

            if args.mode != "just-train":
                res["cutoff"] = cutoff
                res["model"] = model_name
            results.append(res)
            with open(args.output + "-checkpoint", "wb") as fout:
                pickle.dump(results, fout)

    if args.mode == "just-train":
        with open(args.output, "wb") as fout:
            pickle.dump(results, fout)
    else:
        results = pd.concat(results, axis=0).reset_index(drop=True)
        print(results[["model", "cutoff", "roc_auc"]])
        print("Mean AUC:", results.groupby(["model",
                                            "cutoff"])["roc_auc"].mean())

        results.to_pickle(args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
