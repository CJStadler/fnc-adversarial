#!/bin/bash

sudo apt-get install  -y p7zip-full
pip install -r requirements.txt

read "This will remove all results in data, continue?"
rm data/*.csv
rm data/*.zip
echo "export LOKY_PICKLER=pickle" >> ~/.bashrc
echo "set the LOKY_PICKLER environment variable before executing run_parallel.py"
echo "export LOKY_PICKLER=pickle" 

python filter_correct.py

pushd uclmr
echo load | python pred.py
popd

LOKY_PICKLER=pickle python run_parallel.py --help
