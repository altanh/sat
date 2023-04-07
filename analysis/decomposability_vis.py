import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import argparse
import glob
import tqdm


sns.set_style("whitegrid")


def get_mtx_info(csv_file, mtx_root):
    # Get the total number of vertices and edges in the corresponding .mtx file
    problem_name = os.path.basename(csv_file).split("_decomposability")[0]
    mtx_file = os.path.join(mtx_root, "{}.mtx".format(problem_name))
    with open(mtx_file, "r") as f:
        for line in f:
            if not line.startswith("%"):
                n, m, nnz = [int(x) for x in line.split()]
                break

    assert n == m
    return n, nnz


def get_statistics(csv_file, mtx_root):
    # get problem name
    problem_name = os.path.basename(csv_file).split("_decomposability")[0]

    # get vertices and edges first
    n, nnz = get_mtx_info(csv_file, mtx_root)

    df = pd.read_csv(csv_file)

    nontrivial_components = df[df["edges"] > 0]
    num_nontrivial_components = len(nontrivial_components)
    nontrivial_vertex_count = nontrivial_components["vertices"].sum()
    nontrivial_edge_count = nontrivial_components["edges"].sum()
    nontrivial_vertex_var = nontrivial_components["vertices"].var()
    nontrivial_edge_var = nontrivial_components["edges"].var()

    num_trivial_components = len(df[df["edges"] == 0])

    return {
        "problem_name": problem_name,
        "n": n,
        "nnz": nnz,
        "num_nontrivial_components": num_nontrivial_components,
        "nontrivial_vertex_count": nontrivial_vertex_count,
        "nontrivial_edge_count": nontrivial_edge_count,
        "nontrivial_vertex_var": nontrivial_vertex_var,
        "nontrivial_edge_var": nontrivial_edge_var,
        "num_trivial_components": num_trivial_components,
    }


def compute_dataset_statistics(csv_root, mtx_root):
    # get all .csv files
    csv_files = glob.glob(os.path.join(csv_root, "*.csv"))

    # compute statistics for all .csv files
    data = []
    for csv_file in tqdm.tqdm(csv_files):
        data.append(get_statistics(csv_file, mtx_root))

    return pd.DataFrame(data)


# Note for future: there is no "edge loss" in the sense I originally thought.


def compute_features(args):
    # get all .csv files
    csv_files = glob.glob(os.path.join(args.csv_root, "*.csv"))

    nnzs, tot_components = get_components(args, csv_files, singletons=False)
    _, tot_singletons = get_components(args, csv_files, singletons=True)

    vcs = get_vertex_count(csv_files, args.mtx_root)

    log_nnzs = np.log(nnzs)
    log_tot_components = np.log(tot_components)
    log_tot_singletons = np.log(tot_singletons)
    log_vcs = np.log(vcs)


def plot_components(args):
    # get all .csv files
    csv_files = glob.glob(os.path.join(args.csv_root, "*.csv"))

    nnzs, tot_components = get_components(args, csv_files, args.singletons)
    vcs = get_vertex_count(csv_files, args.mtx_root)

    avg_comp_size = np.array(nnzs) / np.array(tot_components)
    avg_vc = np.array(vcs) / np.array(tot_components)

    if args.singletons:
        X = vcs
        Y = tot_components
        title = "Singleton components"
        xlabel = "Original number of vertices"
        ylabel = "Number of singleton components after decomposition"
    else:
        X = vcs
        Y = avg_vc
        title = "Average component size"
        xlabel = "Original number of vertices"
        ylabel = "Average component vertex count after decomposition"

    # plot
    plt.figure()
    if args.hard_file != "":
        hard = pd.read_csv(args.hard_file)
        hard_hashes = hard["hash"].values
        hard_hashes = set(hard_hashes)
        hard_indices = []
        for csv_file in csv_files:
            hsh = os.path.basename(csv_file).split("-")[0]
            hard_indices.append(hsh in hard_hashes)
        hard_indices = np.array(hard_indices)
        plt.scatter(
            X[np.logical_not(hard_indices)],
            Y[np.logical_not(hard_indices)],
            marker=".",
            label="Easier",
        )
        plt.scatter(X[hard_indices], Y[hard_indices], marker="x", label="Hard")
    else:
        plt.scatter(X, Y)

    # plot y = x using axline
    plt.axline((0, 0), slope=1, color="black", linestyle="--", label="y = x")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.title(title)

    # save
    plt.savefig(args.output, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_root", help="Root directory of .csv files")
    parser.add_argument("mtx_root", help="Root directory of .mtx files")
    parser.add_argument("-o", "--output", help="Output file", default="decomp_vc.png")
    parser.add_argument("--hard-file", type=str, default="")
    parser.add_argument(
        "--load-stats",
        type=str,
        default="decomp_stats.csv",
        help="Load stats from file. If the file does not exist, it will be created.",
    )
    args = parser.parse_args()

    if args.load_stats != "" and os.path.exists(args.load_stats):
        df = pd.read_csv(args.load_stats)
    else:
        df = compute_dataset_statistics(args.csv_root, args.mtx_root)
        if args.load_stats != "":
            df.to_csv(args.load_stats, index=False)

    breakpoint()

