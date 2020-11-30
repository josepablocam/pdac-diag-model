from argparse import ArgumentParser
from collections import OrderedDict
import os

import numpy as np
import pandas as pd
import tqdm

import clinical_relevant_diagnoses
import extract_features as ef

CUSTOM_BUCKETS = {
    "age": {
        "bins": [50, 60, 65, 70, 75, 80],
        "labels": ["<50", "50-60", "60-65", "65-70", "70-75", "75-80", ">80"]
    },
}


def assign_to_range(bins, labels, values):
    return np.array(labels)[np.searchsorted(bins, values, side="right")]


def bucket_ages(ages):
    age_bins = CUSTOM_BUCKETS["age"]["bins"]
    age_labels = CUSTOM_BUCKETS["age"]["labels"]
    bucketed = assign_to_range(
        age_bins,
        age_labels,
        ages,
    )
    return bucketed


def sort_buckets(df, column, labels):
    df["dummy"] = df[column].map(lambda x: labels.index(x))
    df = df.sort_values("dummy", ascending=True)
    return df.reset_index(drop=True).drop(columns=["dummy"])


PCT_INFO_STR = "{:.0f} ({:.1f})"


def add_pct_info(df, group_cts):
    df = df.copy()
    for col, denom_ct in group_cts.items():
        df[col] = df[col].map(
            lambda x: PCT_INFO_STR.format(x, (x / float(denom_ct)) * 100))
    return df


def relabel_race_categories(c):
    c = c.upper()
    if any(e in c for e in (
            "NOT RECORDED",
            "UNKNOWN",
            "UNAVAILABLE",
    )):
        return "unknown"
    elif any(e in c for e in (
            "HAWAIIAN",
            "ASIAN PACIFIC",
    )):
        return "pacific islander"
    elif any(e in c for e in ("AMERICAN INDIAN", )):
        return "american indian"
    elif any(e in c for e in ("ASIAN AMERICAN INDIAN", )):
        return "asian indian"
    elif any(e in c for e in (
            "ASIAN",
            "ORIENTAL",
    )):
        return "asian"
    elif any(e in c for e in (
            "HISPANIC",
            "DOMINICAN",
    )):
        return "latino"
    elif any(e in c for e in ("MIDDLE EASTERN", )):
        return "middle eastern"
    elif any(e in c for e in ("WHITE", "CAUCASIAN", "EUROPEAN")):
        return "white"
    elif any(e in c for e in (
            "AFRICAN AMERICAN",
            "BLACK",
    )):
        return "black"
    elif any(e in c for e in ("NATIVE AMERICAN")):
        return "native american"
    elif "Other" in c:
        return "other"
    else:
        return None


def label_race(race):
    ethnic_desc = pd.Series(race).fillna("Unknown")
    ethnic_desc = ethnic_desc.map(relabel_race_categories)
    return ethnic_desc.fillna("other")


def pivot_counts(df, group_map, stat):
    df = df.copy()
    df["group"] = df.UID.map(lambda x: group_map[x])
    cts = df.groupby(["group", stat]).size().to_frame("ct").reset_index()
    pivoted = pd.pivot_table(cts, index=stat, columns="group", values="ct")
    pivoted.columns.name = None
    pivoted = pivoted.reset_index()
    pivoted = pivoted.fillna(0.0)
    pivoted = pivoted.rename(columns={stat: "value"})
    return pivoted


def demographics(df, group_map):
    df["age"] = bucket_ages(df["orig_age"].values)
    df["race"] = label_race(df["orig_race"].values)

    age = pivot_counts(df, group_map, "age")
    age = sort_buckets(age, "value", CUSTOM_BUCKETS["age"]["labels"])
    age["stat"] = "Age"

    gender = pivot_counts(df, group_map, "orig_gender")
    gender["stat"] = "Gender"

    race = pivot_counts(df, group_map, "race")
    race["stat"] = "Race"

    combined = pd.concat([age, gender, race], axis=0)
    return combined


DISEASE_LABELS = {
    "diabetes_mellitus": "diabetes mellitus",
    "diabetes_mellitus_with_complications": "diabetes mellitus",
    "family_history_pancreas_cancer": "family history pancreatic cancer",
    "emphysema": "emphysema",
    "asthma": "asthma",
    "stroke": "stroke",
    "coronary_heart_disease": "coronary heart disease",
    "atherosclerosic_heart_disease": "atherosclerosic heart disease",
    "angina_pectoris": "angina pectoris",
    "ulcer": "ulcer",
    "essential_hypertension": "essential hypertension",
    "chronic_pancreatitis": "chronic pancreatitis",
    "abdominal_pain": "abdominal pain",
    "calculus_gallbladder": "calculus gallbladder",
    "chest_pain": "chest pain",
    "jaundice": "jaundice",
}


def diagnoses(df, group_map):
    codes = clinical_relevant_diagnoses.get_canonical_codes()
    codes = {v: l for v, l in codes.items() if l in DISEASE_LABELS.keys()}
    # at any time
    df = df[df.diag_cd.isin(codes.keys())].copy()
    df["disease_label"] = df["diag_cd"].map(codes).map(DISEASE_LABELS)
    df = df.groupby(["UID", "disease_label"]).head(1)
    diag_cts = pivot_counts(df, group_map, "disease_label")
    diag_cts["stat"] = "Diagnoses"
    return diag_cts


def build_stats_table(demo_path, diag_path, add_pct):
    # IMPORTANT: note that the cutoff is zero days
    diag_df = pd.read_feather(diag_path)
    demo_df = pd.read_feather(demo_path)
    # only for same set of patients
    # since we filter etc
    demo_df = demo_df[demo_df.UID.isin(diag_df.UID.unique())].reset_index()
    status_df = diag_df.groupby("UID")[["UID", "cancer"]].head(1)
    group_map = {
        u: "cancer" if cancer_status else "control"
        for u, cancer_status in zip(status_df.UID, status_df.cancer)
    }
    demo_counts = demographics(
        demo_df, group_map)[["stat", "value", "cancer", "control"]]
    diag_counts = diagnoses(diag_df,
                            group_map)[["stat", "value", "cancer", "control"]]
    acc_df = pd.concat((demo_counts, diag_counts),
                       axis=0).reset_index(drop=True)

    acc_df["value"] = acc_df["value"].str.title()
    acc_df_latex = acc_df.copy()

    if add_pct:
        group_counts = {
            "cancer": status_df.cancer.sum(),
            "control": (~status_df.cancer).sum(),
        }
        acc_df_latex = add_pct_info(acc_df_latex, group_counts)

    acc_df_latex = acc_df_latex.rename(
        columns={
            "stat": "Stat",
            "value": "Value",
            "cancer": "Cases",
            "control": "Controls",
        })
    return acc_df_latex


def get_args():
    parser = ArgumentParser(
        description="Build zero-cutoff clinical features table for writeup")
    parser.add_argument(
        "--demographics",
        type=str,
        nargs="+",
        help="Demographics data",
    )
    parser.add_argument(
        "--diagnoses",
        type=str,
        nargs="+",
        help="Diagnoses data",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output path for latex table",
    )
    parser.add_argument(
        "--add_pct",
        action="store_true",
        help="Add percentage info to latex output",
    )
    parser.add_argument(
        "--column_prefix",
        nargs="+",
        help="Optional column prefix",
    )
    return parser.parse_args()


def merge_summary_tables(tbls, add_pct, prefixes=None):
    result = None
    for ix, tbl in enumerate(tbls):
        if prefixes is not None:
            prefix = prefixes[ix]
            new_cols = [
                prefix + "_" + c if c not in ["Stat", "Value"] else c
                for c in tbl.columns
            ]
            tbl.columns = new_cols
        if result is None:
            result = tbl
        else:
            result = pd.merge(result, tbl, how="left", on=["Stat", "Value"])
            filler_str = "0 (0.0)" if add_pct else "0"
            result = result.fillna(filler_str)
    return result


def main():
    args = get_args()
    summaries = []
    for demo_tbl, diags_tbl in zip(args.demographics, args.diagnoses):
        summary_tbl = build_stats_table(
            demo_tbl,
            diags_tbl,
            args.add_pct,
        )
        summaries.append(summary_tbl)

    if len(summaries) > 0:
        acc_df = merge_summary_tables(
            summaries,
            args.add_pct,
            args.column_prefix,
        )
    else:
        acc_df = summaries[0]

    acc_df.to_latex(
        args.output,
        index=False,
        float_format="%.0f",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
