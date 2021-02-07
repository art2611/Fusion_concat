#!/bin/sh
read -e -p "Enter the fusion type (early/layerX/unimodal) : " FUSION
echo $FUSION
read -e -p "Enter the reid type wanted (BtoB/TtoT/VtoV) : " REID
echo $REID
if [ "$FUSION" =  "unimodal" ]
then
    read -e -p "Enter the trained model type (VtoV / TtoT) : " TRAINED
    echo $TRAINED
else
    TRAINED="BtoB"
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

read -e -p "Enter the fuse type (sum/cat/cat_channel/none) of the trained model :" FUSE
echo FUSE

python test.py --fusion=$FUSION --dataset=$DATASET --reid=$REID --trained=$TRAINED --fuse=$FUSE ;

