import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import seaborn as sns


sns.set_style("whitegrid")


def plot_ratio_vs_nnz(df, use_log_scale=False):
    # plot ratio of clauses to variables vs. number of non-zero entries
    plt.figure()
    plt.scatter(df["num_clauses"] / df["num_vars"], df["nnz"])
    plt.xlabel("Ratio of clauses to variables")
    plt.ylabel("Number of non-zero entries in factor graph")
    if use_log_scale:
        plt.xscale("log")
        plt.yscale("log")
    plt.title("Ratio of clauses to variables vs. number of non-zero entries")
    out_file = "ratio_vs_nnz_log.png" if use_log_scale else "ratio_vs_nnz.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")


def plot_ratio_vs_density(df, use_log_scale=False):
    # plot ratio of clauses to variables vs. density
    plt.figure()
    V = df["num_vars"] + df["num_clauses"]
    E = df["nnz"]
    undirected_density = 2 * E / (V * (V - 1))
    directed_density = E / (V * (V - 1))
    plt.scatter(df["num_clauses"] / df["num_vars"], undirected_density, label="undirected")
    plt.xlabel("Ratio of clauses to variables")
    plt.ylabel("Factor graph density")
    if use_log_scale:
        plt.xscale("log")
        plt.yscale("log")
    plt.title("Ratio of clauses to variables vs. density")
    plt.legend()
    out_file = "ratio_vs_density_log.png" if use_log_scale else "ratio_vs_density.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")


def plot_density_histogram(df):
    # plot density histogram
    V = df["num_vars"] + df["num_clauses"]
    E = df["nnz"]
    undirected_density = 2 * E / (V * (V - 1))
    directed_density = E / (V * (V - 1))
    plt.figure()
    plt.hist(undirected_density, bins=100, label="undirected")
    plt.xlabel("Factor graph density")
    plt.ylabel("Number of factor graphs")
    plt.title("Factor graph density histogram")
    plt.legend()
    plt.savefig("density_hist.png", dpi=300, bbox_inches="tight")


def plot_nnz_histogram(df):
    # plot nnz histogram
    plt.figure()
    plt.hist(df["nnz"], bins=100)
    plt.xlabel("Number of non-zero entries in factor graph")
    plt.ylabel("Number of factor graphs")
    plt.title("Number of non-zero entries in factor graph histogram")
    plt.savefig("nnz_hist.png", dpi=300, bbox_inches="tight")


def plot_total_nnz_cumulative(df):
    # plot cumulative number of non-zero entries
    plt.figure()
    # sort by number of non-zero entries
    df = df.sort_values(by="nnz")
    plt.plot(np.arange(len(df)), np.cumsum(df["nnz"]))
    plt.yscale("log")
    plt.xlabel("Number of factor graphs")
    plt.ylabel("Total number of non-zero entries in factor graphs")
    plt.title("Cumulative number of non-zero entries in factor graphs")
    plt.savefig("nnz_cumulative.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    dataset_stats = sys.argv[1]

    # read in dataset stats
    df = pd.read_csv(dataset_stats)

    # plot ratio of clauses to variables vs. number of non-zero entries
    plot_ratio_vs_nnz(df, use_log_scale=True)
    plot_ratio_vs_density(df, use_log_scale=True)
    plot_density_histogram(df)
    plot_nnz_histogram(df)
    plot_total_nnz_cumulative(df)

    breakpoint()
