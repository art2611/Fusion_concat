#!/bin/sh
read -e -p "Enter the fusion type (early/layerX/unimodal) : " FUSION
echo $FUSION
read -e -p "Enter the reid type (BtoB) : " REID
echo $REID
read -e -p "Enter the dataset name (sysu/regdb) :" DATASET
echo $DATASET
read -e -p "Enter the the fuse type (sum/cat/none) :" FUSE
echo $FUSE
echo \############################### START FOLD 1 \###############################
python train.py --fusion=$FUSION --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=0;
echo \############################### START FOLD 2 \###############################
python train.py --fusion=$FUSION --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=1;
echo \############################### START FOLD 3 \###############################
python train.py --fusion=$FUSION --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=2;
echo \############################### START FOLD 4 \###############################
python train.py --fusion=$FUSION --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=3;
echo \############################### START FOLD 5 \###############################
python train.py --fusion=$FUSION --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=4;
