import z3
import argparse
import numpy as np


def simple_equality_problem(bits):
    x, y, z = z3.BitVecs("x y z", bits)

    problem = z == x + 10 * y
    return problem


def pythagorean_triple_problem(bits):
    x, y, z = z3.BitVecs("x y z", bits)

    problem = (z * z) == (x * x) + (y * y)
    return problem


def weird_pythagorean_triple_problem(bits):
    x, y, z, w = z3.BitVecs("x y z w", bits)

    problem = (z * z) + (w * w) == (x * x) + (y * y)
    return problem


def fermat_last_problem(bits):
    x, y, z = z3.BitVecs("x y z", bits)

    problem = (z * z * z) == (x * x * x) + (y * y * y)
    return problem


def xor_problem(bits):
    x, y, z = z3.BitVecs("x y z", bits)

    problem = z == x ^ y
    return problem


def sum_of_three_cubes(bits, target=None):
    x, y, z, w = z3.BitVecs("x y z w", bits)

    if target is None:
        target = w

    problem = target == x * x * x + y * y * y + z * z * z
    return problem


def weird_xor_problem(bits):
    x, y, z = z3.BitVecs("x y z", bits)

    problem = z > (x ^ y) * (z * z)
    return problem


def random_arithmetic_problem(bits, num_vars=3, depth=3):
    vs = z3.BitVecs(" ".join(["x{}".format(i) for i in range(num_vars)]), bits)

    cmps = [
        lambda x, y: x < y,
        lambda x, y: x > y,
        lambda x, y: x == y,
        lambda x, y: x != y,
    ]
    ops = [
        lambda x, y: x + y,
        lambda x, y: x - y,
        lambda x, y: x * y,
        lambda x, y: x ^ y,
        lambda x, y: x | y,
        lambda x, y: x & y,
        lambda x, y: x % y,
        lambda x, y: x / y,
    ]

    def gen_expr(depth):
        if depth == 0:
            return vs[np.random.randint(num_vars)]
        else:
            op = ops[np.random.randint(len(ops))]
            return op(gen_expr(depth - 1), gen_expr(depth - 1))

    problem = cmps[np.random.randint(len(cmps))](gen_expr(depth), gen_expr(depth))
    return problem


def bitblast(problem):
    # bitblast
    problem = z3.Tactic("simplify")(problem).as_expr()
    problem = z3.Tactic("bit-blast")(problem).as_expr()
    problem = z3.Tactic("tseitin-cnf")(problem)[0]
    return problem


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=32)
    parser.add_argument("--output", type=str, default="bitblast.cnf")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    problem = sum_of_three_cubes(args.bits, target=52)
    problem_str = str(problem).replace("\n", " ")

    problem = bitblast(problem)

    dimacs = problem.dimacs().split("\n")
    # remove comments in reverse order
    while True:
        assert len(dimacs) > 0
        if dimacs[-1].startswith("c"):
            dimacs.pop()
        else:
            break

    print(problem_str)

    with open(args.output, "w") as f:
        f.write("c {}\n".format(problem_str))
        for line in dimacs:
            f.write(line + "\n")

    if args.render:
        import subprocess

        subprocess.call(["./util/cnf2mtx", args.output])
        mtx_file = args.output + ".mtx"
        subprocess.call(
            ["python", "spectral/eigenvalue.py", mtx_file, "--output-dir", "."]
        )
