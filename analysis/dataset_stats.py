import glob
import sys
import os
import pandas as pd


if __name__ == "__main__":
    dataset_dir = sys.argv[1]
    print("Dataset directory: {}".format(dataset_dir))

    mtx_dir = dataset_dir + "_mtx"
    if not os.path.exists(mtx_dir):
        print("Error: mtx directory does not exist")
        sys.exit(1)

    # Get all cnf files in the dataset directory
    cnf_files = glob.glob(os.path.join(dataset_dir, "*.cnf"))
    print("Number of CNF files: {}".format(len(cnf_files)))

    # Stats schema:
    headers = ["file", "num_vars", "num_clauses", "nnz"]
    entries = []

    for cnf in cnf_files:
        # check for corresponding .mtx file
        mtx = os.path.join(mtx_dir, os.path.basename(cnf) + ".mtx")
        if not os.path.exists(mtx):
            print("warning: mtx file does not exist for {}, skipping...".format(cnf))
            continue

        # get number of variables and clauses
        with open(cnf, "r") as f:
            for line in f:
                if line.startswith("p cnf"):
                    num_vars, num_clauses = line.split()[2:4]
                    break

        # get number of non-zero entries by reading first non-comment line
        with open(mtx, "r") as f:
            for line in f:
                if not line.startswith("%"):
                    nnz = line.split()[2]
                    break

        # convert to ints
        num_vars = int(num_vars)
        num_clauses = int(num_clauses)
        nnz = int(nnz)

        # add to entries
        entries.append([os.path.basename(cnf), num_vars, num_clauses, nnz])

    # write to csv
    df = pd.DataFrame(entries, columns=headers)
    df.to_csv(os.path.join(dataset_dir, "stats.csv"), index=False)
