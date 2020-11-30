#!/usr/bin/env python3

from argparse import ArgumentParser
from collections import defaultdict
import copy
import itertools
import os
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
)
import sklearn.model_selection
import skorch
import numpy as np
import tqdm
from xgboost import XGBClassifier

import extract_features as ef
import nn


def get_lr_params():
    """
    Parameter grid for logistic regression
    """
    params = {
        "penalty": ["l1", "l2"],
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    }
    return params


def get_rf_params():
    """
    Parameter grid for random forest
    """
    params = {
        "n_estimators": [10, 50, 100, 200],
        "criterion": ["gini", "entropy"],
        "max_depth": [10, 50, 100, None],
        "min_samples_leaf": [1, 2],
        "min_samples_split": [2, 3, 4],
    }
    return params


def get_xgb_params():
    """
    Parameter grid for XGBoost
    """
    params = {
        "max_depth": [6, 10, 20],
        "learning_rate": [0.005, 0.01, 0.1],
        "n_estimators": [10, 50, 100, 200],
        "reg_lambda": [0.01, 0.1, 1.0, 10.0],
        "reg_alpha": [0.01, 0.1, 1.0, 10.0],
        "tree_method": ["auto"],
    }
    return params


def get_nn_params():
    """
    Parameter grid for feed forward nn
    """
    # module__<param> notation comes from skorch
    # we start from configuration used by
    # Wazir et al and consider
    # adding drop-out to counteract overfitting where necessary
    params = {
        "module__num_hidden_layers": [3],
        "module__hidden_size": [100],
        "module__activation": ["logistic"],
        "module__dropout": [0.0, 0.4, 0.5, 0.6],
    }
    return params


def get_lr():
    """
    Weighted-objective logistic regression
    """
    model = LogisticRegression(class_weight="balanced")
    return model


def get_baecker():
    """
    Baecker model
    """
    # high penalty for no L2
    return LogisticRegression(
        penalty="l2",
        C=1e42,
        class_weight="balanced",
        max_iter=200,
    )


def get_rf():
    """
    Weighted-objective random forest
    """
    model = RandomForestClassifier(
        verbose=3,
        n_jobs=-1,
        class_weight="balanced",
    )
    return model


def get_xgb():
    """
    XGBoost
    """
    model = XGBClassifier(
        verbosity=3,
        n_jobs=-1,
    )
    return model


def set_xgb_scale_pos_weight(m, w):
    """
    Set XGBoost positive weight scale (i.e. make weighted)
    """
    m.set_params(scale_pos_weight=w)
    return m


def get_nn():
    """
    Feed forward NN
    """
    model = nn.get_nn()
    return model


def sample_dataset(X, y, n, seed):
    """
    Sample n patients from the dataframe (with a rng seed).
    Keeps original ratio of cancer to controls.
    """
    splitter = StratifiedShuffleSplit(
        n_splits=2,
        train_size=int(n),
        random_state=seed,
    )
    split = splitter.split(X, y)
    train_ix, _ = next(split)
    return X[train_ix], y[train_ix]


def get_randomized_configs(params, n_configs, seed=None):
    """
    Sample n_configs from the parameter grid
    """
    if seed is not None:
        np.random.seed(seed)
    randomized_configs = []
    for _ in range(n_configs):
        new_config = {}
        for k, vs in params.items():
            new_config[k] = np.random.choice(vs, 1)
        randomized_configs.append(new_config)
    return randomized_configs


def grid_search(model, X, y, params, cv=3):
    """
    Run grid search for a model over data set (X, y) with parameter grid
    params
    """
    print("Grid Search")
    hyperparam_search = GridSearchCV(
        model,
        param_grid=params,
        n_jobs=-1,
        scoring="roc_auc",
        cv=cv,
        verbose=3,
    )
    hyperparam_search.fit(X, y)
    return hyperparam_search


def get_grid_count(params):
    """
    Count how many unique configs the parameter dictionary implies
    """
    return np.prod([len(p) for p in params.values()])


def sanitize_params_for_grouping(params):
    params = [(k, "None" if v is None else v) for k, v in params.items()]
    return tuple(params)


def params_from_grouping_key(params):
    return {k: (None if v == "None" else v) for k, v in params}


def summarize_cv_history(hist):
    """
    Summarize score across entries for each parameter configuration.
    Sort summary so that best model is at top.
    """
    hist = hist.copy()
    hist["params"] = hist["params"].map(sanitize_params_for_grouping)
    unique_configs = hist["params"].unique()
    params_to_id = {p: _id for _id, p in enumerate(unique_configs)}
    id_to_params = {
        _id: params_from_grouping_key(p)
        for p, _id in params_to_id.items()
    }

    hist["params"] = hist["params"].map(lambda x: params_to_id[x])
    hist_summary = hist.groupby("params")[["score", "y_frac"]].mean()
    hist_summary["count"] = hist.groupby("params").size()

    hist_summary = hist_summary.reset_index()
    hist_summary = hist_summary.sort_values(
        "score",
        ascending=False,
    )
    hist_summary["params"] = hist_summary["params"].map(
        lambda x: id_to_params[x])
    return hist_summary


def get_scale_pos_weight(y):
    """
    Compute ratio of negative to positive (i.e. controls to cancer) patients
    """
    y = y.astype(bool)
    return ((~y).sum()) / y.sum()


def cv_loop(X, y, model, params, n_iters, n_obs_sample, seed):
    """
    Run hyper-parameter search loop for a given model, with
    given parameter grid params. Sample data n_iters times, each
    time with n_obs_sample number of patients. Start with given RNG seed.
    """
    history = []

    for i in tqdm.tqdm(range(n_iters)):
        X_sampled, y_sampled = sample_dataset(X, y, n_obs_sample, seed + i)
        y_frac = y.mean()
        scale_pos_weight = get_scale_pos_weight(y)
        # iterations per configuration for randomized search
        # with this sampled dataset
        cv = 3

        if isinstance(model, XGBClassifier):
            model = set_xgb_scale_pos_weight(model, scale_pos_weight)

        if isinstance(model, skorch.NeuralNetClassifier):
            model.set_params(module__input_size=X.shape[1])
            if X_sampled.dtype != np.float32:
                X_sampled = X_sampled.astype(np.float32)
            if y_sampled.dtype != np.float32:
                y_sampled = y_sampled.astype(np.float32)
        try:
            search = grid_search(model, X_sampled, y_sampled, params, cv=cv)
            iter_results = [{
                "params": p,
                "score": s,
                "y_frac": y_frac
            } for p, s in zip(search.cv_results_["params"],
                              search.cv_results_["mean_test_score"])]
            history.extend(iter_results)
        except Exception as err:
            pass

    history = pd.DataFrame(history)
    history_summary = summarize_cv_history(history)
    best_config = history_summary.iloc[0]["params"]
    best_score = history_summary.iloc[0]["score"]
    return {
        "best_config": best_config,
        "best_score": best_score,
        "summary": history_summary,
        "history": history
    }


def get_model(name):
    """
    Convenience wrapper, return model for given name
    """
    models = {
        "lr": get_lr(),
        "rf": get_rf(),
        "xgb": get_xgb(),
        "nn": get_nn(),
        "baecker": get_baecker(),
    }
    return models[name]


def get_params(name):
    """
    Convenience wrapper for getting parameter grid
    """
    params = {
        "lr": get_lr_params(),
        "rf": get_rf_params(),
        "xgb": get_xgb_params(),
        "nn": get_nn_params(),
    }
    return params[name]


def get_args():
    parser = ArgumentParser(
        description=
        "Tune hyperparameters for models using sampling and random search", )
    parser.add_argument("--input", type=str, help="Input feather dataframe")
    parser.add_argument("--features",
                        type=str,
                        help="Diagnoses to use to derive features")
    parser.add_argument("--models", type=str, nargs="+", help="Models to run")
    parser.add_argument("--cutoffs",
                        type=int,
                        nargs="+",
                        help="Cutoffs to run")
    parser.add_argument("--n_obs_sample",
                        type=int,
                        help="Number of patients to sample for each iteration")
    parser.add_argument("--n_iters",
                        type=int,
                        help="Number of sampling iterations to run")
    parser.add_argument("--max_configs",
                        type=int,
                        help="Max number of configurations (if more, sampled)")
    parser.add_argument("--seed", type=int, help="Reproducibility seed")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to dump results",
    )
    return parser.parse_args()


def main():
    """
    Run search over models and cutoffs, and store
    results and configurations
    """
    args = get_args()
    df = pd.read_feather(args.input)
    # subset of diagnoses to consider
    diags = pd.read_pickle(args.features)

    final_summary = []
    final_detailed = []

    # sample configurations for each model
    configs = {}
    for model_ix, model_name in enumerate(args.models):
        model = get_model(model_name)
        params = get_params(model_name)

        use_random_configs = get_grid_count(params) > args.max_configs
        if use_random_configs:
            params = get_randomized_configs(
                params,
                args.max_configs,
                seed=args.seed + model_ix,
            )
        configs[model_name] = params

    # run CV search with the sampled configurations
    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pkl")
    model_results = defaultdict(lambda: [])
    for cutoff_ix, cutoff in tqdm.tqdm(enumerate(args.cutoffs)):
        df_pruned = df[df.date >= cutoff].reset_index(drop=True)
        X, y = ef.get_matrix_and_y(df_pruned, cutoff, np.inf, diags)

        for model_name in configs.keys():
            model = get_model(model_name)
            params = configs[model_name]
            results = cv_loop(
                X,
                y,
                model,
                params,
                args.n_iters,
                args.n_obs_sample,
                args.seed + cutoff_ix,
            )
            hist = results["history"]
            hist["model"] = model_name
            hist["cutoff"] = cutoff
            model_results[model_name].append(hist)

            with open(checkpoint_path, "wb") as fout:
                # convert defaultdict to dict before pickle
                # else lambda raises error
                pickle.dump(dict(model_results), fout)

    # aggregate and summarize results
    detailed = []
    summaries = []
    for model_name, histories in model_results.items():
        history_df = pd.concat(histories, axis=0).reset_index(drop=True)
        detailed.append(history_df)
        model_summary = summarize_cv_history(history_df)
        model_summary["model"] = model_name
        summaries.append(model_summary)

    detailed_df = pd.concat(detailed, axis=0).reset_index(drop=True)
    summaries_df = pd.concat(summaries, axis=0).reset_index(drop=True)

    detailed_path = os.path.join(args.output_dir, "params-detailed.pkl")
    detailed_df.to_pickle(detailed_path)

    summary_path = os.path.join(args.output_dir, "params-summary.pkl")
    summaries_df.to_pickle(summary_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
