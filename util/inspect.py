import os
import sys
import glob
import tempfile


CNF_DIR = "benchmarks/2021"


if __name__ == "__main__":
    prefix = sys.argv[3]
    kind = sys.argv[1]
    out = sys.argv[2]

    assert kind in ["f", "c", "v"]

    kind_map = {
        "f": "factor",
        "c": "clause",
        "v": "variable",
    }

    files = list(glob.glob(os.path.join(CNF_DIR, prefix + "*.cnf")))

    if len(files) == 0:
        print("No files found")
        sys.exit(1)
    elif len(files) > 1:
        print("Multiple files found")
        sys.exit(1)
    else:
        filename = files[0]

    # make temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # convert cnf to mtx
        outfile = os.path.join(tmpdir, prefix + ".mtx")
        cmd = ["./util/cnf2mtx", "-f", filename, "-o", outfile, "-t", kind_map[kind]]

        print(" ".join(cmd))
        os.system(" ".join(cmd))

        # run spectral embedding
        cmd = ["python", "spectral/eigenvalue.py", outfile, "--output-dir", out]
        print(" ".join(cmd))
        os.system(" ".join(cmd))

