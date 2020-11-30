#!/usr/bin/env python3
from argparse import ArgumentParser

import pandas as pd


def get_args():
    parser = ArgumentParser(
        description="Filter feather dataframes based on UID")
    parser.add_argument("--uids", type=str, help="Pickled UIDs to keep")
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        help="List of feather dataframe paths to process",
    )
    parser.add_argument(
        "--output",
        type=str,
        nargs="+",
        help="List of paths to save down new dataframes",
    )
    return parser.parse_args()


def main():
    args = get_args()
    uids = pd.read_pickle(args.uids)
    for input_path, output_path in zip(args.input, args.output):
        print("Filtering", input_path, ", saving to", output_path)
        df = pd.read_feather(input_path)
        df = df[df["UID"].isin(uids)].reset_index(drop=True)
        df.to_feather(output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
