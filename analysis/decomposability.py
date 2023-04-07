"""
We aim to characterize the "decomposability" of a given SAT instance. The idea
is to split the factor graph into two subgraphs induced by edge polarity; then,
each connected component of the subgraphs can be solved essentially trivially,
as all variables appear with the same polarity.

Highly decomposable problems will have few (large) connected components, while
less decomposable problems will have many (small) connected components.
"""
import numpy as np
import scipy as sp
import argparse
import os
import time
import multiprocessing
import ctypes


def init_pool(A_data, A_indices, A_indptr, labels_mp):
    global A
    global labels

    A = sp.sparse.csr_matrix(
        (A_data, A_indices, A_indptr),
        shape=(len(labels_mp), len(labels_mp)),
        copy=False,
    )

    labels = np.frombuffer(labels_mp, dtype="int32")


def do_work(i):
    indices = np.argwhere(labels == i).flatten()
    A_sub = A[np.ix_(indices, indices)]

    # get the subgraph induced by this connected component
    # A_sub = A[labels == i, :][:, labels == i]
    # count the number of edges in the subgraph
    return A_sub.nnz // 2


# TODO: refactor duplicated code in spectral/eigenvalue.py
def split_components(A, compute_edgecounts=False):
    """
    Split a graph into connected components, and return the size distribution.
    """
    n_components, labels = sp.sparse.csgraph.connected_components(A)
    print("found {} connected components".format(n_components))

    # compute size of connected components efficiently, by counting multiplicity of each label
    label_counts = np.bincount(labels)

    edge_counts = None

    if compute_edgecounts:
        # get the indices of components that have more than 2 vertices
        indices = np.argwhere(label_counts > 2).flatten()

        # make new multiprocessing arrays for A
        A_data_mp = multiprocessing.Array(ctypes.c_int64, A.data, lock=False)
        A_indices_mp = multiprocessing.Array(ctypes.c_int32, A.indices, lock=False)
        A_indptr_mp = multiprocessing.Array(ctypes.c_int32, A.indptr, lock=False)

        # make new multiprocessing arrays for labels
        labels_mp = multiprocessing.Array(ctypes.c_int32, labels, lock=False)

        # initialize edge counts
        edge_counts = np.zeros(n_components, dtype="int64")

        # components with 2 vertices have 1 edge
        edge_counts[label_counts == 2] = 1

        # run workers
        with multiprocessing.Pool(
            initializer=init_pool,
            initargs=[A_data_mp, A_indices_mp, A_indptr_mp, labels_mp],
        ) as pool:
            rest_edge_counts = np.array(pool.map(do_work, indices))

        edge_counts[indices] = rest_edge_counts

    return label_counts, labels, edge_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="path to input matrix market file")
    parser.add_argument("output_dir", type=str, help="path to output directory")

    args = parser.parse_args()

    input_file = args.input_file
    output_dir = args.output_dir

    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # load adjacency matrix
    print("loading adjacency matrix...")
    t = time.time()
    A = sp.io.mmread(input_file).tocsr()
    print("|> loaded {} in {:.2f} seconds".format(input_file, time.time() - t))

    # get the largest connected component
    print("getting largest connected component...")
    t = time.time()
    label_counts, labels, _ = split_components(A)
    largest_component = np.argmax(label_counts)

    # get the subgraph induced by the largest connected component
    A = A[labels == largest_component, :][:, labels == largest_component]
    A = A.tocoo()
    print(
        "|> found largest connected component with {} vertices in {:.2f} seconds".format(
            label_counts[largest_component], time.time() - t
        )
    )

    # split into positive and negative edges
    print("splitting into positive and negative edges...")
    t = time.time()

    # positive edges - filter out negative entries
    pos_indices = np.argwhere(A.data > 0).flatten()
    pos_col = A.col[pos_indices]
    pos_row = A.row[pos_indices]
    pos_data = np.ones(len(pos_indices), dtype=int)
    A_pos = sp.sparse.coo_matrix((pos_data, (pos_row, pos_col)), shape=A.shape).tocsr()

    # negative edges - filter out positive entries
    neg_indices = np.argwhere(A.data < 0).flatten()
    neg_col = A.col[neg_indices]
    neg_row = A.row[neg_indices]
    neg_data = np.full(len(neg_indices), -1, dtype=int)
    A_neg = sp.sparse.coo_matrix((neg_data, (neg_row, neg_col)), shape=A.shape).tocsr()

    print(
        "|> split into positive and negative edges in {:.2f} seconds".format(
            time.time() - t
        )
    )

    headers = ["polarity", "component_id", "vertices", "edges"]
    rows = []

    # get the size distribution of the positive and negative subgraphs
    print("getting size distribution of positive and negative subgraphs...")
    t = time.time()
    pos_label_counts, _, pos_edge_counts = split_components(
        A_pos, compute_edgecounts=True
    )
    neg_label_counts, _, neg_edge_counts = split_components(
        A_neg, compute_edgecounts=True
    )
    print(
        "|> computed size distribution in {:.2f} seconds".format(time.time() - t),
        flush=True,
    )

    for i, (label_count, edge_count) in enumerate(
        zip(pos_label_counts, pos_edge_counts)
    ):
        rows.append(["positive", i, label_count, edge_count])

    for i, (label_count, edge_count) in enumerate(
        zip(neg_label_counts, neg_edge_counts)
    ):
        rows.append(["negative", i, label_count, edge_count])

    # write to file
    print("writing to file...")
    t = time.time()
    problem_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(
        output_dir, "{}_decomposability.csv".format(problem_name)
    )

    with open(output_file, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")

    print("|> wrote to {} in {:.2f} seconds".format(output_file, time.time() - t))
