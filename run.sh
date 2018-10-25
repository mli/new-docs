# source activate mxnet-docs

rm -rf build/_build/
make html

aws s3 sync --delete build/_build/html/ s3://mxnet-new-docs/ --acl public-read
cd build/_build/html/ && python -m http.server

sphinx-autogen *rst -t ../../templates/
