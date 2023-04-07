import scipy as sp
import sklearn.manifold
import sklearn.cluster
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import animation
import seaborn as sns
import os
import time
import pyamg
import argparse


sns.set_style("white")

COMPONENT_CUTOFF = 10
MAX_NUM_COMPONENTS = 10
OUTPUT_DIR = "spectral/renders"


def compute_eigenpairs(A, k=2):
    k = k + 1

    # construct symmetric graph Laplacian
    L = sp.sparse.csgraph.laplacian(A, form="array", copy=False, normed=True).tocsr()

    ml = pyamg.smoothed_aggregation_solver(L)
    X = np.random.rand(L.shape[0], k)
    M = ml.aspreconditioner()

    # get smallest eigenpairs
    w, v = sp.sparse.linalg.lobpcg(L, X, M=M, tol=1e-8, largest=False, maxiter=100000)

    return w[1:], v[:, 1:]


def calculate_edges(A, v, d=2):
    red_lines = []
    green_lines = []

    # iterate over edges using row, col, data fields
    for row, col, data in zip(A.row, A.col, A.data):
        if row < col:
            # green if positive edge else red
            color = "g" if data > 0 else "r"

            start = v[row, :d]
            end = v[col, :d]

            if color == "g":
                green_lines.append([start, end])
            else:
                red_lines.append([start, end])

    return np.array(red_lines), np.array(green_lines)


def plot_vertices(ax, v, d=2, cs=None):
    assert d in [2, 3], "only support 2 or 3 dimensions for rendering"
    if cs is None:
        cs = "k"
    s = 2
    a = 0.5
    m = "."
    if d == 2:
        ax.scatter(v[:, 0], v[:, 1], s=s, marker=m, c=cs, alpha=a)
    else:
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=s, marker=m, c=cs, alpha=a)


def plot_edges(ax, lines, d=2):
    assert d in [2, 3], "only support 2 or 3 dimensions for rendering"
    red_lines, green_lines = lines

    lw = 0.5
    a = 0.5

    if d == 2:
        ax.add_collection(LineCollection(red_lines, colors="r", linewidths=lw, alpha=a))
        ax.add_collection(
            LineCollection(green_lines, colors="g", linewidths=lw, alpha=a)
        )
    else:
        ax.add_collection(
            Line3DCollection(red_lines, colors="r", linewidths=lw, alpha=a)
        )
        ax.add_collection(
            Line3DCollection(green_lines, colors="g", linewidths=lw, alpha=a)
        )


def split_subgraphs(A, labels, cutoff=-1):
    # compute size of connected components efficiently, by counting multiplicity of each label
    label_counts = np.bincount(labels)

    # select components (labels) with at least COMPONENT_CUTOFF vertices
    selected_labels = np.argwhere(label_counts >= cutoff).flatten()

    subgraphs = []
    subgraph_sizes = []

    for i in selected_labels:
        subgraphs.append(A[labels == i, :][:, labels == i])
        subgraph_sizes.append(label_counts[i])

    return subgraphs, subgraph_sizes


def split_components(A):
    """
    Split a graph into connected components, and filter out small components.
    Returns the sizes of each component as well.
    """
    n_components, labels = sp.sparse.csgraph.connected_components(A)
    print("found {} connected components".format(n_components))
    return split_subgraphs(A, labels, cutoff=COMPONENT_CUTOFF)


def get_problem_name(input_file):
    return os.path.splitext(os.path.basename(input_file))[0]


def get_positive_adj(A):
    """
    Compute the positive adjacency matrix for a graph.
    """
    Ap = sp.sparse.csr_matrix(A)
    Ap.data = np.ones_like(Ap.data)
    return Ap


def preprocess_matrix(A, only_largest=False, compute_positive=True):
    """
    Preprocess A by splitting into connected components and computing the positive (simple)
    adjacency matrix for each component. If only_largest is True, only return the largest.
    """
    A_subs, _ = split_components(A)
    A_subs.sort(key=lambda A_sub: A_sub.shape[0], reverse=True)
    if only_largest:
        A_subs = [A_subs[0]]
    if compute_positive:
        Ap_subs = [get_positive_adj(A_sub) for A_sub in A_subs]
    else:
        Ap_subs = None
    A_subs = [A_sub.tocoo() for A_sub in A_subs]
    return A_subs, Ap_subs


def plot_spectral_embedding(A, Ap, output_folder, problem_name, suffix, args):
    d = args.dim

    t = time.time()
    v = sklearn.manifold.spectral_embedding(
        Ap, n_components=d, eigen_solver="lobpcg", norm_laplacian=True
    )
    print("-- calculated embedding in {:.2f} seconds".format(time.time() - t))

    if args.save_eigenvectors:
        # save eigenvectors to file
        output_file = os.path.join(
            output_folder,
            "{}_eigenvectors{}.npy".format(problem_name, suffix),
        )
        np.save(output_file, v)

    if args.no_render:
        return

    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    ax = fig.add_subplot(projection="3d" if d == 3 else None)

    # plot vertex positions
    t = time.time()
    plot_vertices(ax, v, d=d)
    print("-- plotted vertex positions in {:.2f} seconds".format(time.time() - t))

    # plot edges as lines
    t = time.time()
    lines = calculate_edges(A, v, d=d)
    plot_edges(ax, lines, d=d)
    print("-- plotted edges in {:.2f} seconds".format(time.time() - t))

    plt.axis("off")

    # save figure to output folder, get output file name from input file name
    if d == 3:

        def animate(i):
            ax.view_init(elev=10.0, azim=i)
            return (fig,)

        def progress_callback(i, n):
            if i % 10 == 0:
                print("-- rendering frame {} of {}".format(i, n))

        anim = animation.FuncAnimation(
            fig, animate, frames=360, interval=1, blit=True
        )
        output_file = os.path.join(
            output_folder, problem_name + "{}.mp4".format(suffix)
        )
        t = time.time()
        anim.save(output_file, dpi=200, fps=30, progress_callback=progress_callback)
        print(
            "! saved animation to {} in {:.2f} seconds]".format(
                output_file, time.time() - t
            )
        )
    else:
        output_file = os.path.join(
            output_folder, problem_name + "{}.png".format(suffix)
        )
        t = time.time()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(
            "! saved figure to {} in {:.2f} seconds".format(
                output_file, time.time() - t
            )
        )

    plt.clf()


def run(args):
    input_file = args.input_file

    # read matrix market file
    t = time.time()
    A = sp.io.mmread(input_file).asfptype().tocsr()
    print("read matrix in {:.2f} seconds".format(time.time() - t))

    # preprocess matrix
    A_subs, Ap_subs = preprocess_matrix(A)

    for i in range(len(A_subs)):
        print(
            "processing component {} with {} vertices...".format(i, A_subs[i].shape[0])
        )

        problem_name = get_problem_name(input_file)

        # make folder if doesn't exist
        output_folder = args.output_dir
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        A_sub = A_subs[i]
        Ap_sub = Ap_subs[i]

        plot_spectral_embedding(A_sub, Ap_sub, output_folder, problem_name, "_{}".format(i), args)


def run_clustering(args):
    """
    Run spectral clustering on the input graph, and split into the induced subgraphs. Then, lay out
    each subgraph using spectral embedding.
    """
    input_file = args.input_file
    d = args.dim

    # read matrix market file
    t = time.time()
    A = sp.io.mmread(input_file).asfptype().tocsr()
    print("read matrix in {:.2f} seconds".format(time.time() - t))

    # preprocess matrix
    A, Ap = preprocess_matrix(A, only_largest=True)
    A = A[0]
    Ap = Ap[0]

    # cluster
    t = time.time()

    # cs = sklearn.cluster.DBSCAN(n_jobs=-1).fit(v_).labels_
    cs = sklearn.cluster.spectral_clustering(
        Ap,
        n_clusters=args.k,
        eigen_solver="lobpcg",
        assign_labels="cluster_qr",
    )
    print("-- calculated clustering in {:.2f} seconds".format(time.time() - t))
    print("-- (found {} clusters)".format(np.max(cs) + 1))

    # split into subgraphs
    t = time.time()
    A_subs, _ = split_subgraphs(A.tocsr(), cs)
    print("-- split into {} subgraphs in {:.2f} seconds".format(len(A_subs), time.time() - t))

    problem_name = get_problem_name(input_file)

    # make folder if doesn't exist
    output_folder = os.path.join(args.output_dir, problem_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # sort clusters by size (largest first)
    A_subs = sorted(A_subs, key=lambda x: x.shape[0], reverse=True)

    for i in range(len(A_subs)):
        print("-- processing cluster {} with {} vertices...".format(i, A_subs[i].shape[0]))

        A, A_p = preprocess_matrix(A_subs[i], only_largest=True)
        A = A[0]
        A_p = A_p[0]
        
        print("-- selected largest component with {} vertices".format(A.shape[0]))

        plot_spectral_embedding(A, A_p, output_folder, problem_name, "_cluster_{}".format(i), args)


def run_split_polarity(args):
    """
    Run spectral embedding, but split the graph into two graphs of positive and negative edges.
    Chooses the largest component of the original graph, and ignores the rest.
    """

    input_file = args.input_file
    d = args.dim

    # read matrix market file
    t = time.time()
    A = sp.io.mmread(input_file).tocsr()
    print("read matrix in {:.2f} seconds".format(time.time() - t))

    # check connected components
    components, component_sizes = split_components(A)

    # choose largest component
    A = components[np.argmax(component_sizes)].tocoo()
    print("chose largest component with {} vertices".format(A.shape[0]))

    # split into positive and negative edges
    t = time.time()

    # positive edges - filter out negative entries
    pos_indices = np.argwhere(A.data > 0).flatten()
    pos_col = A.col[pos_indices]
    pos_row = A.row[pos_indices]
    pos_data = np.ones(len(pos_indices), dtype=float)
    A_pos = sp.sparse.coo_matrix((pos_data, (pos_row, pos_col)), shape=A.shape).tocsr()

    # negative edges - filter out positive entries
    neg_indices = np.argwhere(A.data < 0).flatten()
    neg_col = A.col[neg_indices]
    neg_row = A.row[neg_indices]
    neg_data = np.full(len(neg_indices), -1, dtype=float)
    A_neg = sp.sparse.coo_matrix((neg_data, (neg_row, neg_col)), shape=A.shape).tocsr()

    print(
        "split into positive and negative edges in {:.2f} seconds".format(
            time.time() - t
        )
    )

    for j, A in enumerate([A_pos, A_neg]):
        print("processing " + ("positive" if j == 0 else "negative") + " subgraph...")

        A_subs = []
        Ap_subs = []

        # split into connected components
        A_subs, _ = split_components(A)
        # sort by largest first
        A_subs.sort(key=lambda A_sub: A_sub.shape[0], reverse=True)

        Ap_subs = [sp.sparse.csr_matrix(A_sub) for A_sub in A_subs]
        for Ap_sub in Ap_subs:
            Ap_sub.data = np.ones_like(Ap_sub.data)
        A_subs = [A_sub.tocoo() for A_sub in A_subs]

        for i in range(len(A_subs)):
            if i >= MAX_NUM_COMPONENTS:
                break

            print("processing component {}...".format(i))

            problem_name = os.path.splitext(os.path.basename(input_file))[0]

            # make folder if doesn't exist
            output_folder = os.path.join(args.output_dir, "split", problem_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            A_sub = A_subs[i]
            Ap_sub = Ap_subs[i]

            t = time.time()
            v = sklearn.manifold.spectral_embedding(
                Ap_sub, n_components=d, eigen_solver="lobpcg", norm_laplacian=True
            )
            # _, v = compute_eigenpairs(Ap_sub, k=2)
            print("  calculated embedding in {:.2f} seconds".format(time.time() - t))

            if args.save_eigenvectors:
                # save eigenvectors to file
                output_file = os.path.join(
                    output_folder,
                    "{}_eigenvectors_{}_{}.npy".format(
                        problem_name, ("pos" if j == 0 else "neg"), i
                    ),
                )
                np.save(output_file, v)

            if args.no_render:
                continue

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(projection="3d" if d == 3 else None)

            # plot vertex positions
            t = time.time()
            plot_vertices(ax, v, d=d)
            print(
                "  plotted vertex positions in {:.2f} seconds".format(time.time() - t)
            )

            # plot edges as lines
            t = time.time()
            lines = calculate_edges(A_sub, v, d=d)
            plot_edges(ax, lines, d=d)
            print("  plotted edges in {:.2f} seconds".format(time.time() - t))

            plt.axis("off")
            plt.tight_layout()

            # save figure to output folder
            if d == 3:

                def animate(i):
                    ax.view_init(elev=10.0, azim=i)
                    return (fig,)

                def progress_callback(i, n):
                    if i % 10 == 0:
                        print("  rendering frame {} of {}".format(i, n))

                anim = animation.FuncAnimation(
                    fig, animate, frames=360, interval=1, blit=True
                )
                output_file = os.path.join(
                    output_folder,
                    problem_name
                    + "_"
                    + ("pos" if j == 0 else "neg")
                    + "_{}.mp4".format(i),
                )
                t = time.time()
                anim.save(output_file, fps=30, progress_callback=progress_callback)
                print(
                    "saved animation to {} in {:.2f} seconds".format(
                        output_file, time.time() - t
                    )
                )
            else:
                output_file = os.path.join(
                    output_folder,
                    problem_name
                    + "_"
                    + ("pos" if j == 0 else "neg")
                    + "_{}.png".format(i),
                )
                t = time.time()
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
                print(
                    "saved figure to {} in {:.2f} seconds".format(
                        output_file, time.time() - t
                    )
                )

            plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="path to input matrix market file")
    parser.add_argument("--dim", type=int, default=2, help="dimension of embedding")
    parser.add_argument(
        "--split", action="store_true", help="split into positive and negative edges"
    )
    parser.add_argument(
        "--cluster", action="store_true", help="Perform spectral clustering"
    )
    parser.add_argument(
        "--assign-labels",
        type=str,
        default="cluster_qr",
        help="How to assign labels to clusters",
    )
    parser.add_argument("--k", type=int, default=2, help="Number of clusters")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--save-eigenvectors", action="store_true")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.no_render and not args.save_eigenvectors:
        print("--no-render and --save-eigenvectors both set, so nothing to do...")
        sys.exit(0)

    # ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.split:
        run_split_polarity(args)
    elif args.cluster:
        run_clustering(args)
    else:
        run(args)
