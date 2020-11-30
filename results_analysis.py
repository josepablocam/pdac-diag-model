#!/usr/bin/env python3

from argparse import ArgumentParser
import glob
import os
import pickle
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
import pandas as pd
import seaborn as sns


def relabel_models(df):
    df = df.copy()
    labels = {
        "baecker": "Clinical-LR (baseline)",
        "lr": "LR",
        "nn": "NN",
    }
    col = "model" if "model" in df.columns else "label"
    df["label"] = df[col].map(lambda x: labels[x])
    return df


def get_colors_and_linestyles(methods):
    NUM_COLORS = len(methods)
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    COLORS = sns.color_palette('husl', n_colors=NUM_COLORS)
    mapped_colors = {}
    mapped_linestyles = {}
    for i, method in enumerate(sorted(methods)):
        mapped_colors[method] = COLORS[i]
        mapped_linestyles[method] = LINE_STYLES[i % NUM_STYLES]
    return mapped_colors, mapped_linestyles


def barplot_metric_with_ci(df, stat, output_dir):
    plot_path = os.path.join(output_dir, "{}_comparison.pdf".format(stat))
    methods = sorted(df.label.unique())
    colors, _ = get_colors_and_linestyles(methods)

    cols = [stat + suffix for suffix in ["", "_ci_lb", "_ci_ub"]]
    df = df.groupby(["cutoff", "label"])[cols].mean()
    df = df.reset_index()
    df_stat = pd.pivot_table(df, index="cutoff", values=stat, columns="label")
    df_stat_lb = pd.pivot_table(
        df,
        index="cutoff",
        values="{}_ci_lb".format(stat),
        columns="label",
    )
    df_stat_ub = pd.pivot_table(
        df,
        index="cutoff",
        values="{}_ci_ub".format(stat),
        columns="label",
    )

    # assemble errors, list of entry per method in same order
    # as df_auc
    # each entry is a list of 2 vectors, first vector is
    # amount to subtract for error bar below, and second vector
    # is amount to add for error bar above
    yerr = []
    for label in df_stat.columns:
        lb_diff = (df_stat[label] - df_stat_lb[label]).values
        ub_diff = (df_stat_ub[label] - df_stat[label]).values
        yerr.append((lb_diff, ub_diff))
    # use_colors = [colors[l] for l in df_stat.columns]
    ax = df_stat.plot(kind="bar", yerr=yerr)
    ax.set_xlabel("Cutoff")
    ax.set_ylabel(stat.upper())
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    try:
        plt.tight_layout()
    except:
        pass
    fig = ax.get_figure()
    fig.savefig(plot_path)
    plt.close(fig)


def plot_curves(plot_curve_fun, df, output_path):
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_path)
    cutoffs = sorted(df.cutoff.unique())
    methods = sorted(df.label.unique())
    colors, linestyles = get_colors_and_linestyles(methods)

    for cutoff in cutoffs:
        fig, ax = plt.subplots(1)
        for method in methods:
            method_results = df[(df.label == method) & (df.cutoff == cutoff)]
            plot_curve_fun(
                method_results,
                ax=ax,
                label=method,
                color=colors[method],
                linestyle=linestyles[method],
            )
        ax.set_title("Cutoff={}".format(cutoff))
        try:
            plt.tight_layout()
        except:
            pass
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()


def plot_auc(df, output_dir):
    df = df.copy()
    df.columns = [c.replace("roc_auc", "auc") for c in df.columns]
    barplot_metric_with_ci(df, "auc", output_dir)


def plot_roc(results, plot_cv=False, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1)
    # reference straight line
    ax.plot([0, 1], [0, 1], linestyle="dashed", alpha=0.8)

    nrows = results.shape[0]
    if nrows > 1:
        # from CV
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        for _, row in results.iterrows():
            if plot_cv:
                ax.plot(row.fpr, row.tpr, alpha=0.4)
            interp_tpr = np.interp(mean_fpr, row.fpr, row.tpr)
            # set first elem to zero
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        # combined
        mean_tpr = np.mean(tprs, axis=0)
        fpr = mean_fpr
        tpr = mean_tpr
    else:
        fpr = results.iloc[0].fpr
        tpr = results.iloc[0].tpr

    if 'label' not in kwargs:
        label = "Mean (AUC={:.2f})".format(results["auc"].mean())

    if 'lw' not in kwargs:
        kwargs['lw'] = 2

    if 'alpha' not in kwargs:
        kwargs['alpha'] = 1.0

    ax.plot(
        fpr,
        tpr,
        **kwargs,
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="best")
    return ax


def plot_roc_curves(df, output_dir):
    roc_curves_path = os.path.join(output_dir, "roc_curves.pdf")
    plot_curves(plot_roc, df, roc_curves_path)


def get_formatted_auc(df):
    formatted = [
        "{:.2f} ({:.2f} - {:.2f})".format(*row) for row in zip(
            df["roc_auc"],
            df["roc_auc_ci_lb"],
            df["roc_auc_ci_ub"],
        )
    ]
    return formatted


def auc_table(df, output_dir):
    tbl = df.groupby(["label", "cutoff"
                      ])["roc_auc_ci_lb", "roc_auc", "roc_auc_ci_ub"].mean()
    tbl = tbl.reset_index()
    tbl.to_csv(os.path.join(output_dir, "roc_auc.csv"), index=False)

    latex_tbl = tbl.copy()
    latex_tbl["formatted"] = get_formatted_auc(latex_tbl)
    latex_tbl = latex_tbl[["label", "cutoff", "formatted"]]
    latex_tbl.columns = ["Model", "Cutoff", "AUC (CI)"]
    latex_tbl.to_latex(os.path.join(output_dir, "roc_auc.tex"), index=False)
    return tbl


def table_wrt_specificity(df, stat):
    formatted = []
    spec_cutoffs = [(0.0, 0.949), (0.95, 0.989), (0.99, 1.0)]

    keep_cols = ["label", "cutoff"]
    for ix, row in df.iterrows():
        row_formatted = {}
        for cutoff in spec_cutoffs:
            col_prefix = "spec_{}_{}_".format(*cutoff)

            stat_lb = row[col_prefix + "{}_lb".format(stat)]
            stat_ub = row[col_prefix + "{}_ub".format(stat)]
            new_value = "({:.4f} - {:.4f})".format(stat_lb, stat_ub)

            if cutoff[0] == 0.0:
                new_col = "<={:.3f}".format(cutoff[1])
                if stat == "sens":
                    new_value = ">={:.2f}".format(stat_lb)
                if stat == "ppv":
                    new_value = "{:.4f}".format(stat_ub)
            elif cutoff[1] == 1.0:
                new_col = ">={:.2f}".format(cutoff[0])
                if stat == "sens":
                    new_value = "<={:.2f}".format(stat_ub)
                if stat == "ppv":
                    new_value = "{:.4f}".format(stat_lb)
            else:
                new_col = "{:.2f}-{:.3f}".format(*cutoff)

            row_formatted[new_col] = new_value

            for col in keep_cols:
                row_formatted[col] = row[col]
        formatted.append(row_formatted)
    formatted_df = pd.DataFrame(formatted)
    other_cols = ["<=0.949", "0.95-0.989", ">=0.99"]
    formatted_df = formatted_df[keep_cols + other_cols]
    formatted_df = formatted_df.sort_values(["cutoff", "label"])
    formatted_df = formatted_df.rename(columns={
        "label": "Model",
        "cutoff": "Cutoff"
    })
    return formatted_df


def table_specificity_sensitivity(df, output_dir):
    cols = [c for c in df.columns if c.startswith("spec") and not ("ci" in c)]
    tbl = df.groupby(["label", "cutoff"])[cols].mean()
    tbl = tbl.reset_index()
    tbl.to_csv(os.path.join(output_dir, "spec_sens.csv"), index=False)

    tbl_paper = table_wrt_specificity(tbl, "sens")
    tbl_paper.to_latex(os.path.join(output_dir, "spec_sens.tex"), index=False)
    return tbl


def table_specificity_ppv(df, output_dir):
    cols = [c for c in df.columns if c.startswith("spec") and not ("ci" in c)]
    tbl = df.groupby(["label", "cutoff"])[cols].mean()
    tbl = tbl.reset_index()
    tbl.to_csv(os.path.join(output_dir, "spec_sens.csv"), index=False)

    tbl_paper = table_wrt_specificity(tbl, "ppv")
    tbl_paper.to_latex(os.path.join(output_dir, "spec_ppv.tex"), index=False)
    return tbl


def pretend_screening_table(df, output_dir):
    cols = ["total_count", "ct_high_risk", "ct_cancer_high_risk"]
    tbl = df.groupby(["label", "cutoff"])[cols].mean().reset_index()
    tbl["screen_ratio"] = tbl["ct_high_risk"] / tbl["total_count"]
    tbl["ppv"] = tbl["ct_cancer_high_risk"] / tbl["ct_high_risk"]
    tbl = tbl[[
        "label",
        "cutoff",
        "total_count",
        "ct_high_risk",
        "ct_cancer_high_risk",
        "screen_ratio",
        "ppv",
    ]]
    tbl = tbl.rename(
        columns={
            "label": "Model",
            "cutoff": "Cutoff",
            "total_count": "Avg. Count Patients",
            "ct_high_risk": "Avg. Count High Risk",
            "ct_cancer_high_risk": "Avg Cancer in High Risk",
            "screen_ratio": "Screen Ratio",
            "ppv": "PPV",
        })
    tbl.to_latex(
        os.path.join(output_dir, "pretend_screening.tex"),
        index=False,
    )
    return tbl


def table_counts(df, output_dir):
    tbl = df.groupby(["label",
                      "cutoff"])["count_known_cancer", "total_count"].sum()
    tbl = tbl.reset_index()
    tbl.to_csv(os.path.join(output_dir, "test_counts.csv"), index=False)
    tbl.to_latex(os.path.join(output_dir, "test_counts.tex"), index=False)
    return tbl


def combined_auc_table(named_auc_tables, output_path):
    result_df = None
    for name, df in named_auc_tables.items():
        formatted_df = df[["label", "cutoff"]]
        formatted_df[name] = get_formatted_auc(df)
        if result_df is None:
            result_df = formatted_df
        else:
            result_df = pd.merge(
                result_df,
                formatted_df,
                how="left",
                on=["label", "cutoff"],
            )
    result_df = result_df.sort_values(["cutoff", "label"])
    result_df = result_df.rename(columns={
        "label": "Model",
        "cutoff": "Cutoff"
    })
    result_df.to_latex(output_path, index=False)
    return result_df


def merge_auc_tables(named_auc_tables):
    dfs = []
    columns = set()
    for name, df in named_auc_tables.items():
        df = df.copy()
        df["experiment"] = name
        if len(columns) == 0:
            columns.update(df.columns)
        else:
            columns = columns.intersection(df.columns)
        dfs.append(df)
    columns = list(columns)
    dfs = [d[columns] for d in dfs]
    combined = pd.concat(dfs, axis=0).reset_index(drop=True)
    return combined


def combined_auc_plot(named_auc_tables, output_path):
    df = merge_auc_tables(named_auc_tables)
    cutoffs = sorted(df.cutoff.unique())
    fig, axes = plt.subplots(len(cutoffs), sharex=True, sharey=True)
    n_labels = df.label.unique().shape[0]
    for ix, c in enumerate(cutoffs):
        df_cutoff = df[df["cutoff"] == c]
        ax = axes[ix]
        sns.barplot(
            data=df_cutoff,
            x="experiment",
            y="roc_auc",
            hue="label",
            ax=ax,
        )
        ax.set_title("Cutoff={} days".format(c))
        ax.set_ylim(ymin=0.5, ymax=1.0)
        ax.set_ylabel("AUC")
        ax.set_xlabel("Experiment")
        if ix == 0:
            ax.legend(ncol=n_labels, title=None, loc="best")
        else:
            ax.get_legend().remove()
    plt.tight_layout()
    fig.savefig(output_path)


def get_args():
    parser = ArgumentParser(description="Results analysis")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "auc-summary"],
        default="mode to run in",
    )
    parser.add_argument(
        "--auc",
        type=str,
        nargs="+",
        help="CSVs with AUC info",
    )
    parser.add_argument(
        "--name",
        type=str,
        nargs="+",
        help="Name to use for columns for each file",
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        help="Input files pickle",
    )
    parser.add_argument("--output", type=str, help="Output directory or file")
    return parser.parse_args()


def main_all(args):
    dfs = []
    columns = set()
    for df_path in args.input:
        df = pd.read_pickle(df_path)
        if len(columns) == 0:
            columns.update(df.columns)
        else:
            columns = columns.intersection(df.columns)
        dfs.append(df)
    columns = list(columns)
    dfs = [d[columns] for d in dfs]
    combined_df = pd.concat(dfs, axis=0).reset_index(drop=True)
    combined_df = relabel_models(combined_df)
    plot_auc(combined_df, args.output)
    plot_roc_curves(combined_df, args.output)
    auc_table(combined_df, args.output)
    table_specificity_sensitivity(combined_df, args.output)
    table_specificity_ppv(combined_df, args.output)
    pretend_screening_table(combined_df, args.output)
    table_counts(combined_df, args.output)


def main_auc(args):
    dfs = {}
    for name, df_path in zip(args.name, args.auc):
        dfs[name] = pd.read_csv(df_path)
    df = merge_auc_tables(dfs)
    df.to_csv(args.output + "-raw.csv", index=False)
    combined_auc_table(dfs, args.output + ".tex")


def main():
    args = get_args()
    if args.mode == "all":
        main_all(args)
    elif args.mode == "auc-summary":
        main_auc(args)
    else:
        raise ValueError("Unknown mode", args.mode)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
