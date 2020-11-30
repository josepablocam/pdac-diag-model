#!/usr/bin/env python3

from argparse import ArgumentParser
import codecs
import pandas as pd


def load_df(path):
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fin:
        df = pd.read_csv(fin)
    return df


def filter_to_verified(df, verified_df):
    """
    Only keep entries for patients with an UID in the verified set
    """
    return df[df.UID.isin(verified_df.UID.values)].reset_index(drop=True)


def modify_bidmc_table(df):
    """
    Preprocess UID to be string, create date column,
    drop irrelevant columns and standardize ICD string.
    Specific to BIDMC data.
    """
    df["UID"] = df["UID"].map(str)
    df = df.rename(columns={"DischargeDateIndex": "date"})
    df = df.drop(columns=["AdmitDateIndex"])
    df["diag_cd"] = df["diag_cd"].map(lambda x: x.strip().upper())
    return df


def modify_phc_tabe(df):
    """
    Preprocess UID to be string, create date column,
    drop irrelevant columns and standardize ICD string.
    Specific to PHC data.
    """
    df["UID"] = df["UID"].map(str)
    df = df.rename(columns={"DiagnosisDateIndex": "date"})
    df["diag_cd"] = df["Code"].map(
        lambda x: str(x).strip().replace(".", "").upper())
    return df


def filter_n_days(df, n):
    """
    Only keep patients with a record at least n days prior to diagnosis
    or older
    """
    # patient must have at least one entry n days before
    max_by_patient = df.groupby("UID").date.max()
    ok_uids = max_by_patient[max_by_patient >= n].index.values
    df = df[df.UID.isin(ok_uids)].reset_index(drop=True)
    return df


def preprocess(
        which,
        controls_path,
        cancer_path,
        n_heuristic_filter,
        output,
        verified_cancer=None,
):
    """
    Preprocess control and cancer patients and create single dataframe,
    dump it to feather file (faster reads)
    """
    tables = []
    is_cancer = [False, True]
    paths = [controls_path, cancer_path]
    for path, cancer_col_value in zip(paths, is_cancer):
        print("Loading and processing", path)
        df = load_df(path)
        if cancer_col_value and verified_cancer is not None:
            verified_cancer = pd.read_csv(verified_cancer)
            df = filter_to_verified(df, verified_cancer)
        if which.lower() == "bidmc":
            df = modify_bidmc_table(df)
        elif which.lower() == "phc":
            df = modify_phc_tabe(df)
        else:
            raise ValueError("Unknown table mode:", which)
        if n_heuristic_filter > 0:
            df = filter_n_days(df, n_heuristic_filter)
        df["cancer"] = cancer_col_value
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
        "--cancer",
        type=str,
        help="Path to file for cancer patients",
    )
    parser.add_argument(
        "--verified_cancer",
        type=str,
        help="UIDs for BIDMC patients with verified cancer (based on registry)"
    )
    parser.add_argument(
        "--controls",
        type=str,
        help="Path to file for control patients",
    )
    parser.add_argument(
        "--n_heuristic_filter",
        type=int,
        help=
        "Filter down to patients with an entry at least n days before diagnosis"
    )
    parser.add_argument("--output", type=str, help="Path to dump feather file")
    return parser.parse_args()


def main():
    args = get_args()
    preprocess(args.which,
               args.controls,
               args.cancer,
               args.n_heuristic_filter,
               args.output,
               verified_cancer=args.verified_cancer)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
