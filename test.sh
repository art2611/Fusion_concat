#!/bin/sh
read -e -p "Enter the fusion type (early/layerX/unimodal) : " FUSION
echo $FUSION
read -e -p "Enter the reid type wanted (BtoB/TtoT/VtoV) : " REID
echo $REID
read -e -p "Enter the trained model type (BtoB) : " TRAINED
echo $REID
read -e -p "Enter the dataset name (sysu/regdb) :" DATASET
echo $DATASET
read -e -p "Enter the fuse type (sum/cat/none) of the trained model :" FUSE
echo FUSE

python test.py --fusion=$FUSION --dataset=$DATASET --reid=$REID --trained=$TRAINED --fuse=$FUSE ;

