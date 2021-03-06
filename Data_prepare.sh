#!/bin/sh
mkdir ../Datasets/RegDB/exp
echo "cp files"
cp dataset_use/SYSU/*.txt ../Datasets/SYSU/exp
#cp dataset_use/TWorld/*.txt ../Datasets/TWorld/exp
cp dataset_use/RegDB/*.txt ../Datasets/RegDB/exp
echo "rm npy files"
rm ../Datasets/SYSU/*.npy
rm ../Datasets/RegDB/*.npy
#rm ../Datasets/TWorld/*.npy
echo "pre_process_regdb_clean"
python pre_process_regdb_clean.py
echo "pre_process_sysu_clean"
python pre_process_sysu_clean.py
#python pre_process_thermalWORLD.py
echo "rm saved models, results and runs"
rm ../save_model/*
rm -r results.txt
rm -r runs


