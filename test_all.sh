#!/bin/sh
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
echo $FUSE

python test.py --fusion="early" --dataset=$DATASET --reid="BtoB" --trained="BtoB" --fuse=$FUSE ;
python test.py --fusion="layer1" --dataset=$DATASET --reid="BtoB" --trained="BtoB" --fuse=$FUSE ;
python test.py --fusion="layer2" --dataset=$DATASET --reid="BtoB" --trained="BtoB" --fuse=$FUSE ;
python test.py --fusion="layer3" --dataset=$DATASET --reid="BtoB" --trained="BtoB" --fuse=$FUSE ;
python test.py --fusion="layer4" --dataset=$DATASET --reid="BtoB" --trained="BtoB" --fuse=$FUSE ;
python test.py --fusion="layer5" --dataset=$DATASET --reid="BtoB" --trained="BtoB" --fuse=$FUSE ;

