#!/usr/env/bin

# NOTE: this risk stratification is based solely on distribution
# properties (percentiles) for our risk score. It does not
# account for medical costs etc, so should not be viewed as
# a recommendation for risk thresholds, but rather just an experiment
# based on our own scores and their distribution
from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

THRESHOLDS_PREFIX = "percentile"


def preprocess_df(df, model="lr"):
    df = df[df["model"] == model]
    df = df.copy()
    df["odds"] = df["yprob"] / (1 - df["yprob"])
    return df


def find_thresholds(df, percentiles):
    named_percentiles = [(THRESHOLDS_PREFIX + '_' + p,
                          lambda x, p=p: np.percentile(x, float(p)))
                         for p in percentiles]
    thresholds = df.groupby("cutoff")["odds"].agg(named_percentiles)
    thresholds = thresholds.reset_index()
    return thresholds


def assign_risk_groups(df, thresholds_df, group_names):
    df = pd.merge(df, thresholds_df, how="left", on="cutoff")
    columns = [
        c for c in thresholds_df.columns if c.startswith(THRESHOLDS_PREFIX)
    ]
    thresh_columns = sorted(columns, key=lambda x: float(x.split("_")[1]))
    group_col = "risk_group"
    df[group_col] = None
    for ix, c in enumerate(thresh_columns):
        df.loc[(df["odds"] < df[c])
               & pd.isnull(df[group_col]), group_col] = group_names[ix]
    # last group
    df.loc[pd.isnull(df[group_col]), group_col] = group_names[-1]
    return df


def load_and_assign_risk_groups(
        train_path,
        test_path,
        risk_percentiles,
        risk_group_names,
        model,
):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = preprocess_df(train_df, model=model)
    test_df = preprocess_df(test_df, model=model)

    thresholds_df = find_thresholds(train_df, percentiles=risk_percentiles)
    test_groups = assign_risk_groups(test_df, thresholds_df, risk_group_names)
    return test_groups


def get_args():
    parser = ArgumentParser(
        description="Distributional risk group information")
    parser.add_argument(
        "--train",
        type=str,
        help="Path to pROC for data to compute thresholds from")
    parser.add_argument("--test",
                        type=str,
                        help="Path to pROC for data to apply thresholds to")
    parser.add_argument("--model", type=str, help="Model to use", default="lr")
    parser.add_argument("--percentiles",
                        type=str,
                        nargs="+",
                        default="Percentile thresholds")
    parser.add_argument("--groups",
                        type=str,
                        nargs="+",
                        help="Groups in ascending name order")
    parser.add_argument("--output", type=str, help="Output directory")
    return parser.parse_args()


def main():
    args = get_args()
    test_groups = load_and_assign_risk_groups(
        args.train,
        args.test,
        args.percentiles,
        args.groups,
        args.model,
    )
    test_groups.to_csv(args.output, index=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
