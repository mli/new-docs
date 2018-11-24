#!/bin/bash
set -e
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64

conda env update -f environment.yml
conda activate mxnet-docs

rm -rf build/_build/
make html

if [[ ${env.BRANCH_NAME} == master ]]; then
    aws s3 sync --delete build/_build/html/ s3://beta.mxnet.io/ --acl public-read
fi
