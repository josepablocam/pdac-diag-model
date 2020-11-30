#!/usr/bin/env python3

from argparse import ArgumentParser
import codecs
import pandas as pd

import extract_features as ef


def load_df(path):
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fin:
        df = pd.read_csv(fin)
    return df


def modify_bidmc_table(df):
    """
    Preprocess age, race and gender for BIDMC data
    """
    df["UID"] = df["UID"].map(str)
    df = df.rename(columns={
        "AgeAsOfFirstDiagnosis": "age",
        "ethnic_desc": "race",
    })
    if "DiagnosisDateIndex" in df.columns:
        df = df.drop(columns=["DiagnosisDateIndex"])
    df["gender"] = df["gender"].map({
        "M": "Male",
        "F": "Female",
        "U": "Uknown",
    })
    df_static = ef.get_static_demo_factors(df)
    return df_static


def modify_phc_table(df):
    """
    Preprocess age, race and gender for PHC data
    """
    df["UID"] = df["UID"].map(str)
    column_map = {
        "Gender": "gender",
        "Race": "race",
        "AgeAsOfFirstDiagnosis": "age",
    }
    if "DiagnosisDateIndex" in df.columns:
        df = df.drop(columns=["DiagnosisDateIndex"])
    df = df.rename(columns=column_map)
    df_static = ef.get_static_demo_factors(df)
    return df_static


def preprocess(
        which,
        paths,
        output,
):
    """
    Preprocess control and cancer patients and create single dataframe,
    dump it to feather file (faster reads)
    """
    tables = []
    is_cancer = [False, True]
    for path in paths:
        print("Loading and processing", path)
        df = load_df(path)
        if which.lower() == "bidmc":
            df = modify_bidmc_table(df)
        elif which.lower() == "phc":
            df = modify_phc_table(df)
        else:
            raise ValueError("Unknown table mode:", which)
        tables.append(df)
    combined = pd.concat(tables, axis=0)
    combined = combined.reset_index(drop=True)
    print("Saving down {} records to {}".format(combined.shape[0], output))
    combined.to_feather(output)


def get_args():
    parser = ArgumentParser(description="Load data and preprocess")
    parser.add_argument(
        "--which",
        choices=[
            "bidmc",
            "phc",
            "BIDMC",
            "PHC",
        ],
        type=str,
        help="Data source",
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        help="Path to input demographic files",
    )
    parser.add_argument("--output", type=str, help="Path to dump feather file")
    return parser.parse_args()


def main():
    args = get_args()
    preprocess(
        args.which,
        args.input,
        args.output,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
