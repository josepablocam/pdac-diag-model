#!/usr/bin/env python
from argparse import ArgumentParser
import os

import numpy as np
import pandas as pd
import tqdm

import icd_r_wrapper


def get_model_weights(model, features, feature_desc_map=None):
    # return
    weighted_features = zip(features, model.coef_[0])
    df = pd.DataFrame(
        list(weighted_features),
        columns=["feature", "weight"],
    )
    if feature_desc_map is not None:
        df["feature_desc"] = df["feature"].map(
            lambda x: feature_desc_map.get(x, None))
    df["odds_ratio"] = np.exp(df["weight"])
    df = df.sort_values("weight", ascending=False)
    return df


def get_feature_explanations(features):
    desc = [
        icd_r_wrapper.get_code_explanation(code)
        for code in tqdm.tqdm(features)
    ]
    return dict(zip(features, desc))


def get_model(models, model_name, cutoff):
    model_map = {(n, c): m for n, c, m in models}
    return model_map[(model_name, cutoff)]


def get_args():
    parser = ArgumentParser(description="Create table with model weights")
    parser.add_argument(
        "--trained_models",
        type=str,
        help="Path to trained model list",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Models to create table for",
    )
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        help="Cutoffs to create table for",
    )
    parser.add_argument(
        "--features",
        type=str,
        help="Path to pickled diagnoses used",
    )
    parser.add_argument(
        "--feature_desc",
        type=str,
        help="Path to saved diagnoses descriptions if any",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory to dump tables",
    )
    return parser.parse_args()


def main():
    args = get_args()
    trained_models = pd.read_pickle(args.trained_models)
    features = pd.read_pickle(args.features)
    if not os.path.exists(args.feature_desc):
        feature_desc_map = get_feature_explanations(features)
        pd.to_pickle(feature_desc_map, args.feature_desc)
    else:
        feature_desc_map = pd.read_pickle(args.feature_desc)

    pd.set_option('display.max_colwidth', -1)
    for model_name in args.models:
        for cutoff in args.cutoffs:
            model = get_model(trained_models, model_name, cutoff)
            df = get_model_weights(
                model,
                features,
                feature_desc_map=feature_desc_map,
            )
            df["model"] = model_name
            df["cutoff"] = cutoff

            df.to_csv(
                os.path.join(args.output,
                             "{}-{}.csv".format(model_name, cutoff)),
                index=False,
            )
            df_latex = df[["feature_desc", "odds_ratio"]]
            df_latex = df_latex.rename(columns={
                "feature_desc": "Diagnosis",
                "odds_ratio": "O.R."
            })
            df_latex.to_latex(
                os.path.join(args.output,
                             "{}-{}.tex".format(model_name, cutoff)),
                index=False,
                float_format="%.3f",
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
