#!/bin/sh
read -e -p "Enter the fusion type (early/layerX/unimodal) : " FUSION
echo $FUSION

REID="BtoB"

read -e -p "Enter the dataset name (sysu/regdb) :" DATASET
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
echo "BEGINING OF THE FIRST TRAINING : EARLY "
echo \############################### START FOLD 1 \###############################
python train.py --fusion="early" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=0;
echo \############################### START FOLD 2 \###############################
python train.py --fusion="early" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=1;
echo \############################### START FOLD 3 \###############################
python train.py --fusion="early" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=2;
echo \############################### START FOLD 4 \###############################
python train.py --fusion="early" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=3;
echo \############################### START FOLD 5 \###############################
python train.py --fusion="early" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=4;
echo "BEGINING OF THE SECOND TRAINING : LAYER1 "
echo \############################### START FOLD 1 \###############################
python train.py --fusion="layer1" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=0;
echo \############################### START FOLD 2 \###############################
python train.py --fusion="layer1" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=1;
echo \############################### START FOLD 3 \###############################
python train.py --fusion="layer1" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=2;
echo \############################### START FOLD 4 \###############################
python train.py --fusion="layer1" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=3;
echo \############################### START FOLD 5 \###############################
python train.py --fusion="layer1" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=4;
echo "BEGINING OF THE FIRST TRAINING : layer2 "
echo \############################### START FOLD 1 \###############################
python train.py --fusion="layer2" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=0;
echo \############################### START FOLD 2 \###############################
python train.py --fusion="layer2" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=1;
echo \############################### START FOLD 3 \###############################
python train.py --fusion="layer2" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=2;
echo \############################### START FOLD 4 \###############################
python train.py --fusion="layer2" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=3;
echo \############################### START FOLD 5 \###############################
python train.py --fusion="layer2" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=4;
echo "BEGINING OF THE SECOND TRAINING : layer3 "
echo \############################### START FOLD 1 \###############################
python train.py --fusion="layer3" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=0;
echo \############################### START FOLD 2 \###############################
python train.py --fusion="layer3" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=1;
echo \############################### START FOLD 3 \###############################
python train.py --fusion="layer3" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=2;
echo \############################### START FOLD 4 \###############################
python train.py --fusion="layer3" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=3;
echo \############################### START FOLD 5 \###############################
python train.py --fusion="layer3" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=4;
echo "BEGINING OF THE FIRST TRAINING : layer4 "
echo \############################### START FOLD 1 \###############################
python train.py --fusion="layer4" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=0;
echo \############################### START FOLD 2 \###############################
python train.py --fusion="layer4" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=1;
echo \############################### START FOLD 3 \###############################
python train.py --fusion="layer4" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=2;
echo \############################### START FOLD 4 \###############################
python train.py --fusion="layer4" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=3;
echo \############################### START FOLD 5 \###############################
python train.py --fusion="layer4" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=4;
echo "BEGINING OF THE SECOND TRAINING : layer5 "
echo \############################### START FOLD 1 \###############################
python train.py --fusion="layer5" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=0;
echo \############################### START FOLD 2 \###############################
python train.py --fusion="layer5" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=1;
echo \############################### START FOLD 3 \###############################
python train.py --fusion="layer5" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=2;
echo \############################### START FOLD 4 \###############################
python train.py --fusion="layer5" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=3;
echo \############################### START FOLD 5 \###############################
python train.py --fusion="layer5" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=4;