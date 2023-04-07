import os
import argparse
import glob
import pandas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="path to data directory")
    parser.add_argument(
        "--output-dir", type=str, default="", help="path to output directory"
    )
    parser.add_argument("--dim", type=int, default=2, help="dimension of embedding")
    parser.add_argument(
        "--split", action="store_true", help="split graph based on polarity"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=-1,
        help="number of problems to visualize (smallest to largest)",
    )
    parser.add_argument(
        "--hard",
        type=str,
        default="",
        help="only visualize hard problems listed by this file",
    )
    parser.add_argument(
        "--no-render", action="store_true", help="do not render embeddings"
    )
    parser.add_argument(
        "--save-eigenvectors", action="store_true", help="save eigenvectors"
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    d = args.dim
    split = args.split
    num = args.num
    hard_file = args.hard
    no_render = args.no_render
    save_eigenvectors = args.save_eigenvectors
    hard_problems = set()

    if hard_file != "":
        hard_problems = pandas.read_csv(hard_file)
        hard_problems = hard_problems["hash"]
        hard_problems = set(hard_problems)

    # get list of input files, sort from smallest to largest filesize
    input_files = glob.glob(os.path.join(data_dir, "*.mtx"))
    input_files.sort(key=lambda x: os.path.getsize(x))

    total_files = len(input_files)
    if len(hard_problems) > 0:
        total_files = len(hard_problems)

    processed = 0
    # run spectral embedding on each input file
    for i, input_file in enumerate(input_files):
        if len(hard_problems) > 0:
            basename = os.path.basename(input_file)
            problem_hash = basename.split("-")[0]
            if problem_hash not in hard_problems:
                continue

        if num > 0 and i >= num:
            break


        cmd = "python spectral/eigenvalue.py {} --dim {}".format(input_file, d)
        if output_dir != "":
            cmd += " --output-dir {}".format(output_dir)
        if split:
            cmd += " --split"
        if no_render:
            cmd += " --no-render"
        if save_eigenvectors:
            cmd += " --save-eigenvectors"

        print(
            "running spectral embedding on {}... ({}/{})".format(
                input_file, processed + 1, total_files
            )
        )
        print("=============================================")
        print("[" + cmd + "]")

        res = os.system(cmd)
        print()

        if res != 0:
            print("error on {}".format(input_file))
            break

        processed += 1
