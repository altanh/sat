#!/bin/bash

DATASET_DIR=./benchmarks/2021_mtx
OUTPUT_DIR=./data/2021_decomposability

for f in $DATASET_DIR/*.mtx; do
    python ./analysis/decomposability.py $f $OUTPUT_DIR
done
