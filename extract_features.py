#!/usr/bin/env python3

import pandas as pd
import numpy as np
import multiprocessing as mp
import tqdm


def is_feature_map(elem):
    return isinstance(elem, dict)


def diags_from_feature_map(feat_to_diags):
    return [code for group in feat_to_diags.values() for code in group]


def get_list_of_diag_codes(diags):
    if is_feature_map(diags):
        feat_map = diags
        diags = diags_from_feature_map(feat_map)
        num_features = len(feat_map.keys())
        return [d for ds in feat_map.values() for d in ds]
    else:
        return diags


def get_matrix_and_y(df, start_date, end_date, diags, return_uids=False):
    """
    Extract features from a dataframe based on list of diagnosis codes
    or a dictionary of a feature label to diagnosis codes (i.e. feature map)
    Return matrix X of covariates and vector y of cancer status.
    """
    if is_feature_map(diags):
        feat_map = diags
        diags = diags_from_feature_map(feat_map)
        num_features = len(feat_map.keys())
        feat_ix = {
            d: ix
            for ix, group in enumerate(feat_map.values()) for d in group
        }
    else:
        num_features = len(diags)
        feat_ix = {d: ix for ix, d in enumerate(diags)}

    subset_df = df[(df.date >= start_date) & (df.date <= end_date)]
    subset_df = subset_df[subset_df.diag_cd.isin(diags)].reset_index(drop=True)
    cts = subset_df.groupby(["UID", "diag_cd"]).size().to_frame(name='ct')
    cts = cts.reset_index()
    # TODO: don't think we need this here, do we?
    cts = cts[cts.diag_cd.isin(diags)]
    cts["ct"] = 1.0

    df["UID"] = df["UID"].astype(str)
    unique_uids = list(sorted(df["UID"].unique()))
    cancer_status = df.groupby("UID")["UID", "cancer"].head(1)
    cancer_status = dict(zip(cancer_status.UID, cancer_status.cancer))

    # start with empty matrix, all zeros and fill in
    feats = np.zeros((len(unique_uids), num_features), dtype=np.float32)
    uid_ix = {u: ix for ix, u in enumerate(unique_uids)}

    for (u, d) in tqdm.tqdm(list(zip(cts.UID, cts.diag_cd))):
        feats[uid_ix[u], feat_ix[d]] = 1

    y = np.array([cancer_status[u] for u in unique_uids])
    if return_uids:
        # sorted so that they match the order of rows in feats
        uids = [u for u, _ in sorted(uid_ix.items(), key=lambda x: x[1])]
        return (feats, y), uids
    else:
        return feats, y


def relabel_race(race):
    """
    Standardize race to Baecker terms
    """
    if pd.isnull(race):
        return "Unknown"
    race = race.lower()
    if "white" in race:
        return "White"
    elif "black" in race or "african american" in race:
        return "Black"
    elif "hispanic" in race:
        return "Other"
    elif "unknown" in race or "not recorded" in race or "declined" in race:
        return "Unknown"
    else:
        return "Other"


def create_age_buckets(ages_vec, cutoff):
    """
    Return bucketed 5-year age groups, after adjusting age (in years)
    for cutoff (in days)
    """
    min_age = 0
    max_age = 105
    age_step = 5
    age_buckets = np.arange(min_age, max_age + age_step, age_step).tolist()
    age_buckets = [-np.inf] + age_buckets + [np.inf]
    adj_ages_vec = (ages_vec * 365 - cutoff) / 365.0
    bucketed_age = pd.cut(adj_ages_vec, age_buckets)
    age_df = pd.get_dummies(bucketed_age)
    age_df.columns = age_df.columns.astype(str)
    return age_df


def get_static_demo_factors(df):
    """
    Extract Baecker demographic factors
    male/female, white/black/other/unknown.
    We add age dynamically to adjust for prediction window.
    """
    gender_df = pd.get_dummies(
        df["gender"].astype("category",
                            categories=["Male", "Female", "Unknown"]),
        prefix="gender",
    )
    gender_df = gender_df[["gender_Male"]]
    relabeled_race = df["race"].map(relabel_race)
    race_df = pd.get_dummies(
        relabeled_race.astype(
            "category", categories=["White", "Black", "Other", "Unknown"]),
        prefix="race",
    )
    df = df.copy()
    df.columns = ["orig_" + c if c != "UID" else c for c in df]
    return pd.concat((df, gender_df, race_df), axis=1)


def get_X_y(df):
    """
    Return X and y. X is the df without columns UID and cancer,
    and y is the cancer column.
    """
    y = df["cancer"]
    df.drop(columns=["UID", "cancer"], inplace=True)
    return df, y


def get_min_ct_diags(df, n):
    """
    Return diagnoses codes that are observed at least n times in dataframe df
    """
    cts = df.groupby("diag_cd").size()
    cts = cts.sort_values(ascending=False)
    return cts[cts >= n].index.values
