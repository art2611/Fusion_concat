#!/bin/sh
mkdir ../Datasets/RegDB/exp
cp dataset_use/SYSU/*.txt ../Datasets/SYSU/exp
#cp dataset_use/TWorld/*.txt ../Datasets/TWorld/exp
cp dataset_use/RegDB/*.txt ../Datasets/RegDB/exp
rm ../Datasets/SYSU/*.npy
rm ../Datasets/RegDB/*.npy
#rm ../Datasets/TWorld/*.npy
python pre_process_regdb_clean.py
python pre_process_sysu_clean.py
#python pre_process_thermalWORLD.py
rm ../save_model/*
rm -r results.txt
rm -r runs


