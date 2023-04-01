import os
import argparse
import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="path to data directory")
    parser.add_argument("--dim", type=int, default=2, help="dimension of embedding")
    parser.add_argument("--split", action="store_true", help="split graph based on polarity")
    parser.add_argument("--num", type=int, default=-1, help="number of problems to visualize (smallest to largest)")
    args = parser.parse_args()

    data_dir = args.data_dir
    d = args.dim
    split = args.split
    num = args.num

    # get list of input files, sort from smallest to largest filesize
    input_files = glob.glob(os.path.join(data_dir, "*.mtx"))
    input_files.sort(key=lambda x: os.path.getsize(x))

    # run spectral embedding on each input file
    for i, input_file in enumerate(input_files):
        if num > 0 and i >= num:
            break

        print("running spectral embedding on {}".format(input_file))
        if split:
            os.system("python util/eigenvalue.py {} --dim {} --split".format(input_file, d))
        else:
            os.system("python util/eigenvalue.py {} --dim {}".format(input_file, d))
