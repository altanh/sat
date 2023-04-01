import scipy as sp
import sklearn.manifold
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


def plot_vertices(ax, v, d=2):
    assert d in [2, 3], "only support 2 or 3 dimensions for rendering"
    if d == 2:
        ax.scatter(v[:, 0], v[:, 1], s=1, marker=".", c="k", alpha=0.5)
    else:
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=1, marker=".", c="k", alpha=0.5)


def plot_edges(ax, lines, d=2):
    assert d in [2, 3], "only support 2 or 3 dimensions for rendering"
    red_lines, green_lines = lines

    if d == 2:
        ax.add_collection(
            LineCollection(red_lines, colors="r", linewidths=0.5, alpha=0.5)
        )
        ax.add_collection(
            LineCollection(green_lines, colors="g", linewidths=0.5, alpha=0.5)
        )
    else:
        ax.add_collection(
            Line3DCollection(red_lines, colors="r", linewidths=0.5, alpha=0.5)
        )
        ax.add_collection(
            Line3DCollection(green_lines, colors="g", linewidths=0.5, alpha=0.5)
        )


def run(args):
    COMPONENT_CUTOFF = 10

    input_file = args.input_file
    d = args.dim

    # read matrix market file
    t = time.time()
    A = sp.io.mmread(input_file).asfptype().tocsr()
    print("read matrix in {:.2f} seconds".format(time.time() - t))

    # check connected components
    t = time.time()
    n_components, labels = sp.sparse.csgraph.connected_components(A)
    print(
        "found {} connected components in {:.2f} seconds".format(
            n_components, time.time() - t
        )
    )

    A_subs = []
    Ap_subs = []

    # split into connected components
    for i in range(n_components):
        num_vertices = np.sum(labels == i)
        print("- component {} has {} vertices".format(i, num_vertices))
        if num_vertices < COMPONENT_CUTOFF:
            print("  (skipping...)")
            continue

        # get submatrix
        A_sub = A[labels == i, :][:, labels == i]
        # convert to positive adjacency matrix
        Ap_sub = sp.sparse.csr_matrix(A_sub)
        Ap_sub.data = np.ones_like(Ap_sub.data)
        A_subs.append(A_sub.tocoo())
        Ap_subs.append(Ap_sub)

    for i in range(len(A_subs)):
        print("processing component {}...".format(i))

        A_sub = A_subs[i]
        Ap_sub = Ap_subs[i]

        t = time.time()
        v = sklearn.manifold.spectral_embedding(
            Ap_sub, n_components=d, eigen_solver="lobpcg", norm_laplacian=True
        )
        # _, v = compute_eigenpairs(Ap_sub, k=2)
        print("calculated embedding in {:.2f} seconds".format(time.time() - t))

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d" if d == 3 else None)

        # plot vertex positions
        t = time.time()
        plot_vertices(ax, v, d=d)
        print("plotted vertex positions in {:.2f} seconds".format(time.time() - t))

        # plot edges as lines
        t = time.time()
        lines = calculate_edges(A_sub, v, d=d)
        plot_edges(ax, lines, d=d)
        print("plotted edges in {:.2f} seconds".format(time.time() - t))

        plt.axis("off")

        # save figure to output folder, get output file name from input file name
        problem_name = os.path.splitext(os.path.basename(input_file))[0]

        if d == 3:

            def animate(i):
                ax.view_init(elev=10.0, azim=i)
                return (fig,)

            def progress_callback(i, n):
                if i % 10 == 0:
                    print("rendering frame {} of {}".format(i, n))

            anim = animation.FuncAnimation(
                fig, animate, frames=360, interval=1, blit=True
            )
            output_file = os.path.join(
                "spectral/renders", problem_name + "_{}.mp4".format(i)
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
                "spectral/renders", problem_name + "_{}.png".format(i)
            )
            t = time.time()
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(
                "saved figure to {} in {:.2f} seconds".format(
                    output_file, time.time() - t
                )
            )

        plt.clf()


def split_components(A):
    """
    Split a graph into connected components, and filter out small components.
    Returns the sizes of each component as well.
    """
    n_components, labels = sp.sparse.csgraph.connected_components(A)
    components = []
    component_sizes = []
    print("found {} connected components".format(n_components))

    # compute size of connected components efficiently, by counting multiplicity of each label
    label_counts = np.bincount(labels)

    # select components (labels) with at least COMPONENT_CUTOFF vertices
    selected_labels = np.argwhere(label_counts >= COMPONENT_CUTOFF).flatten()

    for i in selected_labels:
        components.append(A[labels == i, :][:, labels == i])
        component_sizes.append(label_counts[i])

    return components, component_sizes


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
    t = time.time()
    components, component_sizes = split_components(A)
    print(
        "found {} connected components in {:.2f} seconds".format(
            len(components), time.time() - t
        )
    )

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

            A_sub = A_subs[i]
            Ap_sub = Ap_subs[i]

            t = time.time()
            v = sklearn.manifold.spectral_embedding(
                Ap_sub, n_components=d, eigen_solver="lobpcg", norm_laplacian=True
            )
            # _, v = compute_eigenpairs(Ap_sub, k=2)
            print("calculated embedding in {:.2f} seconds".format(time.time() - t))

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(projection="3d" if d == 3 else None)

            # plot vertex positions
            t = time.time()
            plot_vertices(ax, v, d=d)
            print("plotted vertex positions in {:.2f} seconds".format(time.time() - t))

            # plot edges as lines
            t = time.time()
            lines = calculate_edges(A_sub, v, d=d)
            plot_edges(ax, lines, d=d)
            print("plotted edges in {:.2f} seconds".format(time.time() - t))

            plt.axis("off")

            # save figure to output folder, get output file name from input file name
            problem_name = os.path.splitext(os.path.basename(input_file))[0]

            # make folder if doesn't exist
            output_folder = os.path.join("spectral/renders/split", problem_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            if d == 3:

                def animate(i):
                    ax.view_init(elev=10.0, azim=i)
                    return (fig,)

                def progress_callback(i, n):
                    if i % 10 == 0:
                        print("rendering frame {} of {}".format(i, n))

                anim = animation.FuncAnimation(
                    fig, animate, frames=360, interval=1, blit=True
                )
                output_file = os.path.join(
                    output_folder,
                    problem_name
                    + "_"
                    + ("pos" if j == 0 else "neg")
                    + "_{}.gif".format(i),
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
    args = parser.parse_args()

    if args.split:
        run_split_polarity(args)
    else:
        run(args)
