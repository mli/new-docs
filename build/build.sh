#!/bin/bash
set -e
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64

conda env update -f environment.yml
conda activate mxnet-docs

rm -rf build/_build/
make html
