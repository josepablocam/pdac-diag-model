#!/usr/bin/env python3
from argparse import ArgumentParser
import pickle

import pandas as pd
import numpy as np


def demo_to_match_df(df, field=None):
    """
    Group demographics info by age/gender creating grouped UIDs
    """
    df_matches = df.groupby(["gender", "age"])["UID"].apply(list)
    if field is None:
        field = "patients"
    return df_matches.to_frame(name=field).reset_index()


def sample_controls(cancer_ct, control_uids, ratio):
    """
    Randomly (w/o replacement) sample ratio * cancer_ct from control_uids
    """
    return np.random.choice(
        control_uids,
        size=int(np.ceil(cancer_ct * ratio)),
        replace=False,
    )


def compute_exact_control_matches(demo_cancer, demo_control):
    """
    Sample controls based on exact age/sex matching. This is a naive
    algorithm and we could likely increase the number of controls if
    we matched differently, but not clear that there is much benefit
    from going down that road.
    """
    cancer_uids = demo_to_match_df(demo_cancer, field="cancer_UID")
    control_uids = demo_to_match_df(demo_control, field="control_UID")
    combined = pd.merge(
        cancer_uids,
        control_uids,
        how="left",
        on=["gender", "age"],
    )
    assert not combined["control_UID"].isnull().any(
    ), "Missing exact match -- naive matching fails"
    combined["cancer_ct"] = combined["cancer_UID"].map(len)
    combined["control_ct"] = combined["control_UID"].map(len)
    combined["sample_ratio"] = combined["control_ct"] / combined["cancer_ct"]
    sample_ratio = combined["sample_ratio"].min()
    print("Matched case/controls sampling ratio of {} for controls".format(
        sample_ratio))
    combined["sampled_UID"] = [
        sample_controls(cancer_ct, control_UID, sample_ratio)
        for cancer_ct, control_UID in zip(
            combined["cancer_ct"],
            combined["control_UID"],
        )
    ]
    control_uids = [uid for grp in combined["sampled_UID"] for uid in grp]
    return control_uids


def get_args():
    parser = ArgumentParser(description="Control/case matching for BIDMC")
    parser.add_argument("--cancer", type=str, help="BIDMC cancer demographics")
    parser.add_argument(
        "--controls",
        type=str,
        help="BIDMC controls demographics",
    )
    parser.add_argument(
        "--filtered_diagnosis_data",
        type=str,
        help="BIDMC diagnosis .feather file with filtered patients etc",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for pickled list of all UIDs to keep for BIDMC",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="RNG seed",
    )
    return parser.parse_args()


def main():
    args = get_args()
    df_control = pd.read_csv(args.controls)
    df_control["UID"] = df_control["UID"].astype(str)

    df_cancer = pd.read_csv(args.cancer)
    df_cancer = df_cancer.rename(columns={"AgeAsOfFirstDiagnosis": "age"})
    df_cancer["UID"] = df_cancer["UID"].astype(str)

    ok_uids = set(pd.read_feather(args.filtered_diagnosis_data).UID.unique())
    df_control = df_control[df_control["UID"].isin(ok_uids)].reset_index(
        drop=True)
    df_cancer = df_cancer[df_cancer["UID"].isin(ok_uids)].reset_index(
        drop=True)

    if args.seed:
        np.random.seed(args.seed)

    sampled_control_uids = compute_exact_control_matches(df_cancer, df_control)
    final_uids = sampled_control_uids + df_cancer["UID"].values.tolist()
    with open(args.output, "wb") as fout:
        pickle.dump(final_uids, fout)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
