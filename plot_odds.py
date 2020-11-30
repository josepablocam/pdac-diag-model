#!/usr/bin/env python3
from argparse import ArgumentParser

import pandas as pd
import seaborn as sns


def prepare_data(df):
    df = df.copy()
    df["odds"] = df["yprob"] / (1 - df["yprob"])
    df["group"] = df["ytrue"].map(lambda x: "case" if x else "control")
    return df


def plot_odds(df, xcutoff):
    g = sns.FacetGrid(data=df, col="cutoff", hue="group")
    g.map(sns.distplot, "odds", kde_kws={"cumulative": True})
    for ax in g.axes.flatten():
        ax.set_xlim(0, xcutoff)
    g.set_titles("{col_name} days")
    g.set_xlabels("Risk Score")
    g.set_ylabels("Fraction of Population")
    g.add_legend()
    return g


def get_args():
    parser = ArgumentParser(description="Plot odds and cdf")
    parser.add_argument("--input", type=str, help="Path to pROC.csv data")
    parser.add_argument("--model", type=str, help="Model to plot")
    parser.add_argument("--cutoff", type=float, help="Score cutoff for plot limits")
    parser.add_argument("--output", type=str, help="Output path for plot")
    return parser.parse_args()


def main():
    args = get_args()
    df = pd.read_csv(args.input)
    df = df[df["model"] == args.model]
    df = prepare_data(df)
    p = plot_odds(df, args.cutoff)
    p.savefig(args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
