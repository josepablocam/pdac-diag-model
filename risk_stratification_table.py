#!/usr/bin/env python3

from argparse import ArgumentParser
import pandas as pd


def float_format(v, min_value=0.02):
    if v < min_value:
        return "{:.4f}".format(v)
    elif v >= 1000:
        v_1k = float(v) / 1000.0
        if v_1k == int(v_1k):
            return "{}k".format(int(v_1k))
        else:
            return "{:.2f}k".format(v_1k)
    elif v == int(v):
        return str(int(v))
    else:
        return "{:.2f}".format(v)


def create_table(df):
    key_cols = [
        ("risk_group", "Group"),
        ("cutoff", "Cutoff"),
    ]
    stat_cols = [
        ("cancer", "Num. Cases"),
        ("controls", "Num. Control"),
        ("rate", "Frac. with Cases"),
        ("sensitivity", "Frac. of All Cases"),
        ("specificity", "Frac. of All Controls"),
    ]
    df = df[df["stat"].isin([k for k, _ in stat_cols])]
    df = df.copy()

    # adjust specifity so that it is actually fraction of all controls
    df["lb_final"] = df["lb"]
    df["ub_final"] = df["ub"]

    df.loc[df["stat"] == "specificity", "value"] = 1 - df[
        df["stat"] == "specificity"]["value"]
    # note that UB/LB are flipped
    df.loc[df["stat"] == "specificity", "lb_final"] = 1 - df[
        df["stat"] == "specificity"]["ub"]
    df.loc[df["stat"] == "specificity", "ub_final"] = 1 - df[
        df["stat"] == "specificity"]["lb"]

    value_strs = []
    for _, row in df.iterrows():
        if row.lb_final == row.ub_final:
            value_strs.append(float_format(row.value))
        else:
            s = "{} ({}-{})".format(
                float_format(row.value),
                float_format(row.lb_final),
                float_format(row.ub_final),
            )
            value_strs.append(s)
    df["value_str"] = value_strs

    df = df[["cutoff", "risk_group", "stat", "value_str"]]
    pv = pd.pivot_table(
        df,
        index=["cutoff", "risk_group"],
        columns="stat",
        values="value_str",
        aggfunc="first",
    )
    pv = pv.reset_index()
    pv.columns.name = None
    pv = pv.rename(columns=dict(key_cols + stat_cols))
    return pv


def get_args():
    parser = ArgumentParser(
        description="Create latex table for risk strat stats")
    parser.add_argument("--input",
                        type=str,
                        help="Input csv file with risk strat stats")
    parser.add_argument("--output", type=str, help="Output latex")
    return parser.parse_args()


def main():
    args = get_args()
    df = pd.read_csv(args.input)
    pv = create_table(df)
    pv.to_latex(args.output, index=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
