#!/bin/bash
# check Jenkinsfile for a sample usage
set -ex

if [ $# -ne 1 ]; then
    echo "Usage: $0 TARGET_DIR"
    exit -1
fi

DIR=$1

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64

cd python

conda activate mxnet-docs

rm -rf build/_build/
make html

rm -rf ../$DIR
mv build/_build/html ../$DIR
