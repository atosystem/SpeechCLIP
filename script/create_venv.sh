#!/bin/bash

# How to execute?
#   bash script/create_venv.sh

python3 -m venv venv
source venv/bin/activate
export PIP_USER=false
pip install --upgrade pip --no-cache-dir
