#! /bin/bash
set -e

#echo "Experiment MBE -----------"
#echo "Preprocessing Data...."
#python 41_preprocess.py 
#echo "Training ..."
#python 42_train.py
#echo "Done ---------------------------------------------------------------"
#
#sleep 60
#
#echo "Experiment STAI -----------"
#echo "Preprocessing Data...."
#python stai_preprocess.py 
#echo "Training ..."
#python stai_train.py
#echo "Done ---------------------------------------------------------------"
#
#sleep 60
#
echo "Experiment MBE + STAI -----------"
echo "Preprocessing Data...."
python data_preprocess.py 
echo "Training ..."
python data_train.py
echo "Done ---------------------------------------------------------------"