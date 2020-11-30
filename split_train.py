#!/usr/env/bin python3
from argparse import ArgumentParser
import os

import pandas as pd
import sklearn.model_selection


def split_data(df, test_size, seed):
    """
    Split train and test patients, keep imbalance constant in
    train and test splits.
    """
    splitter = sklearn.model_selection.StratifiedShuffleSplit(
        2,
        test_size=test_size,
        random_state=seed,
    )
    uid_status = df.groupby("UID")[["UID", "cancer"]].head(1)
    split = splitter.split(uid_status, uid_status.cancer.values)
    train_ixs, test_ixs = next(split)

    train_uids = uid_status.iloc[train_ixs].UID.values
    test_uids = uid_status.iloc[test_ixs].UID.values

    df_train = df[df.UID.isin(train_uids)].reset_index(drop=True)
    df_test = df[df.UID.isin(test_uids)].reset_index(drop=True)
    return df_train, df_test


def get_args():
    parser = ArgumentParser(description="Split data into train/test")
    parser.add_argument("--input", type=str, help="Input feather dataframe")
    parser.add_argument("--test_size", type=float, help="Fraction for test")
    parser.add_argument("--seed", type=int, help="RNG seed")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    return parser.parse_args()


def main():
    args = get_args()
    df = pd.read_feather(args.input)
    train_df, test_df = split_data(df, args.test_size, args.seed)

    n_train = len(train_df.UID.unique())
    n_test = len(test_df.UID.unique())

    base_file_name = os.path.basename(args.input)
    base_file_name, _ = os.path.splitext(base_file_name)

    print("{} patients for train".format(n_train))
    train_df.to_feather(
        os.path.join(args.output_dir, base_file_name + "-train.feather"),
    )

    print("{} patients for test".format(n_test))
    test_df.to_feather(
        os.path.join(args.output_dir, base_file_name + "-test.feather"),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
