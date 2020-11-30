#!/usr/bin/env python3

from argparse import ArgumentParser
import os

import pandas as pd
import seaborn as sns


def plot_empirical_cdf(df, field):
    df = df.copy()
    df["cancer"] = df["cancer"].map(lambda x: "Case" if x else "Control")
    df = df.rename(columns={"cancer": "group"})
    g = sns.FacetGrid(df, hue="group")
    g.map(sns.distplot, field)
    g.add_legend()
    return g


def compute_code_counts(df):
    # only care about unique counts since features are binary
    cts = df.groupby(["UID", "cancer"])["diag_cd"].agg(lambda x: len(set(x)))
    cts = cts.to_frame(name="ct")
    cts = cts.reset_index()
    return cts


def compute_time_intervals(df):
    time = df.groupby(["UID", "cancer"])["date"].agg(lambda x: max(x) - min(x))
    time = time.to_frame(name="time_interval")
    time = time.reset_index()
    return time


def get_args():
    parser = ArgumentParser(
        description=
        "Plot distribution of unique codes counts and time interval (i.e. history length)"
    )
    parser.add_argument("--input", type=str, help="Input feather dataframe")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to output directory",
    )
    return parser.parse_args()


def main():
    args = get_args()
    df = pd.read_feather(args.input)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    df_codes = compute_code_counts(df)
    ax_codes = plot_empirical_cdf(df_codes, "ct")
    ax_codes.set_xlabels("# Diagnosis Codes")
    ax_codes.set_ylabels("Empirical CDF (patients)")
    codes_plot_path = os.path.join(args.output_dir, "code_ct_distribution.pdf")
    ax_codes.savefig(codes_plot_path)

    df_time = compute_time_intervals(df)
    ax_time = plot_empirical_cdf(df_time, "time_interval")
    ax_time.set_xlabels("Time interval of medical history (days)")
    ax_time.set_ylabels("Empirical CDF (patients)")
    time_plot_path = os.path.join(args.output_dir,
                                  "time_interval_distribution.pdf")
    ax_time.savefig(time_plot_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
