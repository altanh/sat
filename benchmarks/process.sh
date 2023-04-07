#!/bin/bash

cd 2021

# Decompress each .cnf.xz file in the 2021 directory.
for f in *.cnf.xz; do
    echo "Decompressing $f..."
    xz -d $f
done

# Convert each .cnf file to the .mtx factor graph encoding.
for f in *.cnf; do
    echo "Converting $f to .mtx..."
    ../../util/cnf2mtx -f $f -o ${f}.mtx -t factor
done

cd .. # Back to the benchmarks directory.
mkdir 2021_mtx
mv 2021/*.mtx 2021_mtx
