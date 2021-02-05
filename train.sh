#!/bin/sh
read -e -p "Enter the fusion type (early/layerX/unimodal) : " FUSION
echo $FUSION
echo "BONJOUR"
if [ "$FUSION" =  "unimodal" ]
then
    read -e -p "Enter the wanted reid type (VtoV / TtoT) : " REID
    echo $VAR
else
    REID="BtoB"
fi
read -e -p "Enter the dataset name (SYSU/RegDB) :" DATASET
echo $DATASET
if [ "$DATASET" =  "regdb" ]
then
  DATASET="RegDB"
elif [ "$DATASET" =  "sysu" ]
then
  DATASET="SYSU"
fi

read -e -p "Enter the the fuse type (sum/cat/cat_channel/none) :" FUSE
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
