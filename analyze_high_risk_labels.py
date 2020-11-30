#!/usr/bin/env python3

from argparse import ArgumentParser

import pandas as pd

import risk_stratification
import extract_features


def annotate_with_diagnoses_used(risk_df, diags_df, cutoff, codes):
    # dates within their test date
    pruned_diags_df = diags_df[diags_df.date >= cutoff]
    pruned_diags_df["UID"] = pruned_diags_df["UID"].astype(str)
    # now only diagnoses we considered when modeling
    pruned_diags_df = pruned_diags_df[pruned_diags_df["diag_cd"].isin(codes)]
    # add what outcome we have for each
    pruned_diags_df = pd.merge(pruned_diags_df, risk_df, how="inner", on="UID")
    return pruned_diags_df


def annotate_with_model_weights_info(diags_df, model_weights_df):
    # sort in descending order
    model_weights_df = model_weights_df.sort_values("odds_ratio",
                                                    ascending=False)
    model_weights_df["feature"] = model_weights_df["feature"].astype(str)
    sorted_codes = model_weights_df["feature"].astype(str).values.tolist()

    # code position: position of diagnosis code in codes
    # sorted by model weights (i.e. lower index -> more highly associated
    # with PDAC diagnosis in our model)
    sorted_codes_map = {code: ix for ix, code in enumerate(sorted_codes)}
    diags_df["code_position"] = diags_df["diag_cd"].map(
        lambda c: sorted_codes_map[c])

    # code description and weight
    diags_df = pd.merge(
        diags_df,
        model_weights_df[[
            "feature",
            "feature_desc",
            "weight",
        ]],
        how="left",
        left_on="diag_cd",
        right_on="feature",
    )
    return diags_df


def get_args():
    parser = ArgumentParser(
        description=
        "Take test-data risk group predictions and add observed diagnosis info"
    )
    parser.add_argument(
        "--risk_train_path",
        type=str,
        help="Risk pROC data (train)",
    )
    parser.add_argument(
        "--risk_test_path",
        type=str,
        help="Risk pROC data (test) -- will add diagnosis info here",
    )
    parser.add_argument(
        "--risk_percentiles",
        type=float,
        nargs="+",
        help="Risk group percentiles",
    )
    parser.add_argument(
        "--risk_group_names",
        type=str,
        nargs="+",
        help="Risk group names (aligned with --risk_percentiles)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name to filter test risk data",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        help="Cutoff to filter test risk data",
    )
    parser.add_argument(
        "--diagnoses",
        type=str,
        help="Path to .feather diagnoses data",
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        help="Path to .csv features (diagnoses) used by model and weights",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output .csv",
    )
    return parser.parse_args()


def main():
    args = get_args()
    risk_df = risk_stratification.load_and_assign_risk_groups(
        args.risk_train_path,
        args.risk_test_path,
        args.risk_percentiles,
        args.risk_group_names,
        args.model_name,
    )
    risk_df = risk_df[risk_df["cutoff"] == args.cutoff]
    risk_df = risk_df.drop(columns=["percentile_75", "percentile_99"])
    high_risk_df = risk_df[risk_df["risk_group"] == "High"].copy()

    df_diags = pd.read_feather(args.diagnoses)

    model_weights_df = pd.read_csv(args.model_weights)
    model_weights_df["feature"] = model_weights_df["feature"].astype(str)

    high_risk_df = annotate_with_diagnoses_used(
        high_risk_df,
        df_diags,
        args.cutoff,
        model_weights_df["feature"].values.tolist(),
    )
    high_risk_df = annotate_with_model_weights_info(
        high_risk_df,
        model_weights_df,
    )
    # for each patient/code entry take single entry
    high_risk_df = high_risk_df.groupby(["UID", "diag_cd"]).head(1)
    # have highest confidence predictions first
    high_risk_df = high_risk_df.sort_values(
        ["yprob", "UID", "weight"],
        ascending=[False, True, False],
    )
    # clean up column names
    high_risk_df = high_risk_df.drop(columns=[
        "feature",
        "Code",
        "cancer",
        "yprob",
        "date",
    ])
    clean_col_names = {
        "diag_cd": "Diagnosis_code",
        "model": "Model",
        "cutoff": "Cutoff",
        "ytrue": "Cancer",
        "odds": "Risk_score",
        "risk_group": "Risk_group",
        "code_position": "Diagnosis_code_position",
        "feature_desc": "Diagnosis_code_description",
        "weight": "Diagnosis_code_model_weight",
    }
    high_risk_df = high_risk_df.rename(columns=clean_col_names)
    high_risk_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
