#!/bin/sh
read -e -p "Enter the fusion type (early/layerX) : " FUSION
echo $FUSION
read -e -p "Enter the reid type (BtoB) : " REID
echo $REID
read -e -p "Enter the dataset name (sysu/regdb) :" DATASET
echo $DATASET

python train.py --fusion=$FUSION --dataset=$DATASET --reid=$REID --fold=0;
python train.py --fusion=$FUSION --dataset=$DATASET --reid=$REID --fold=1;
python train.py --fusion=$FUSION --dataset=$DATASET --reid=$REID --fold=2;
python train.py --fusion=$FUSION --dataset=$DATASET --reid=$REID --fold=3;
python train.py --fusion=$FUSION --dataset=$DATASET --reid=$REID --fold=4;
