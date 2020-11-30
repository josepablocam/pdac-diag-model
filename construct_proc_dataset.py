#!/usr/bin/env python3

from argparse import ArgumentParser
import pandas as pd


def create_proc_dataframe(df):
    acc = []
    for _, row in df.iterrows():
        model = row.model
        cutoff = row.cutoff
        ytrue = row.y_true
        yprob = row.y_prob
        if "UID" in df.columns:
            uids = row.UID
            if isinstance(uids, float):
                uids = [None] * len(ytrue)
        else:
            uids = [None] * len(ytrue)
        entries = [(model, cutoff, uid, yt, yp) for uid, yt, yp in zip(uids, ytrue, yprob)]
        acc.extend(entries)
    acc_df = pd.DataFrame(acc, columns=["model", "cutoff", "UID", "ytrue", "yprob"])
    return acc_df


def get_args():
    parser = ArgumentParser(description="Produce csv for pROC tests")
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        help="Input files",
    )
    parser.add_argument("--output", type=str, help="Output csv")
    return parser.parse_args()


def main():
    args = get_args()
    dfs = []
    for path in args.input:
        df = pd.read_pickle(path)
        dfs.append(df)
    combined_df = pd.concat(dfs, axis=0)
    proc_df = create_proc_dataframe(combined_df)
    proc_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
