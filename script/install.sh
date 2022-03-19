#!/bin/bash

# How to execute?
#   bash script/install.sh

export PIP_USER=false

mkdir -p tmp_tools
cd tmp_tools

git clone --depth 1 https://github.com/s3prl/s3prl
cd s3prl
pip install --editable ./ || exit 1
cd ..

git clone --depth 1 https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./ || exit 1
cd ..

cd ..

pip install -r requirements.txt

pip uninstall dtw-python -y
pip uninstall pesq -y
