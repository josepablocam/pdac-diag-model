#!/usr/bin/env python3

from argparse import ArgumentParser

import pickle
import pandas as pd
import extract_features as ef


def get_args():
    parser = ArgumentParser(description="Get diagnoses with at least n counts")
    parser.add_argument("--input", type=str, help="Path to feather dataframe")
    parser.add_argument("--min_count",
                        type=int,
                        help="Minimum count of diagnosis")
    parser.add_argument("--output",
                        type=str,
                        help="Output path to pickle diagnosis")
    return parser.parse_args()


def main():
    args = get_args()
    df = pd.read_feather(args.input)
    diags = ef.get_min_ct_diags(df, args.min_count)
    with open(args.output, "wb") as fout:
        pickle.dump(diags, fout)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
