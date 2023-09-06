#! /bin/bash
set -e

echo "Experiment 2 -----------"
echo "Preprocessing Data...."
python 41_preprocess.py 
echo "Training ..."
python 42_train.py
echo "Done ---------------------------------------------------------------"