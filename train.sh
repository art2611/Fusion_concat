#!/bin/sh
read -e -p "Enter the fusion type (early/layerX/fc_fuse/gmu/unimodal) : " FUSION
echo $FUSION
echo "BONJOUR"
if [ "$FUSION" =  "unimodal" ]
then
    read -e -p "Enter the wanted reid type (VtoV / TtoT) : " REID
    echo $VAR
else
    REID="BtoB"
fi
read -e -p "Enter the dataset name (SYSU/RegDB/TWorld) :" DATASET
echo $DATASET
if [ "$DATASET" =  "regdb" ]
then
  DATASET="RegDB"
elif [ "$DATASET" =  "sysu" ]
then
  DATASET="SYSU"
fi

read -e -p "Enter the the fuse type (sum/cat/cat_channel/fc_fuse/gmu/none) :" FUSE
echo $FUSE

read -e -p "Enter the the LOO needed (Query / Gallery) :" LOO
echo $LOO

for i in `seq 0 4`;
  do
          echo "BEGINING OF THE TRAINING : $j - Fold = $i fuse = $FUSE"
          python train.py --fusion=$FUSION --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=$i --LOO=$LOO;
  done