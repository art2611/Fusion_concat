#!/bin/sh

REID="BtoB"

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
echo "BEGINING OF THE TRAINING : EARLY "
echo \############################### START FOLD 1 EARLY \###############################
python train.py --fusion="early" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=0;
echo \############################### START FOLD 2 EARLY \###############################
python train.py --fusion="early" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=1;
echo \############################### START FOLD 3 EARLY \###############################
python train.py --fusion="early" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=2;
echo \############################### START FOLD 4 EARLY \###############################
python train.py --fusion="early" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=3;
echo \############################### START FOLD 5 EARLY \###############################
python train.py --fusion="early" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=4;
echo "BEGINING OF THE SECOND TRAINING : LAYER1 "
echo \############################### START FOLD 1 LAYER1 \###############################
python train.py --fusion="layer1" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=0;
echo \############################### START FOLD 2 LAYER1 \###############################
python train.py --fusion="layer1" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=1;
echo \############################### START FOLD 3 LAYER1 \###############################
python train.py --fusion="layer1" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=2;
echo \############################### START FOLD 4 LAYER1 \###############################
python train.py --fusion="layer1" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=3;
echo \############################### START FOLD 5 LAYER1 \###############################
python train.py --fusion="layer1" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=4;
echo "BEGINING OF THE FIRST TRAINING : LAYER2 "
echo \############################### START FOLD 1 LAYER2 \###############################
python train.py --fusion="layer2" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=0;
echo \############################### START FOLD 2 LAYER2 \###############################
python train.py --fusion="layer2" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=1;
echo \############################### START FOLD 3 LAYER2 \###############################
python train.py --fusion="layer2" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=2;
echo \############################### START FOLD 4 LAYER2 \###############################
python train.py --fusion="layer2" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=3;
echo \############################### START FOLD 5 LAYER2 \###############################
python train.py --fusion="layer2" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=4;
echo "BEGINING OF THE SECOND TRAINING : LAYER3 "
echo \############################### START FOLD 1 LAYER3 \###############################
python train.py --fusion="layer3" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=0;
echo \############################### START FOLD 2 LAYER3 \###############################
python train.py --fusion="layer3" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=1;
echo \############################### START FOLD 3 LAYER3 \###############################
python train.py --fusion="layer3" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=2;
echo \############################### START FOLD 4 LAYER3 \###############################
python train.py --fusion="layer3" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=3;
echo \############################### START FOLD 5 LAYER3 \###############################
python train.py --fusion="layer3" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=4;
echo "BEGINING OF THE FIRST TRAINING : LAYER4 "
echo \############################### START FOLD 1 LAYER4 \###############################
python train.py --fusion="layer4" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=0;
echo \############################### START FOLD 2 LAYER4 \###############################
python train.py --fusion="layer4" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=1;
echo \############################### START FOLD 3 LAYER4 \###############################
python train.py --fusion="layer4" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=2;
echo \############################### START FOLD 4 LAYER4 \###############################
python train.py --fusion="layer4" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=3;
echo \############################### START FOLD 5 LAYER4 \###############################
python train.py --fusion="layer4" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=4;
echo "BEGINING OF THE SECOND TRAINING : LAYER5 "
echo \############################### START FOLD 1 LAYER5 \###############################
python train.py --fusion="layer5" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=0;
echo \############################### START FOLD 2 LAYER5 \###############################
python train.py --fusion="layer5" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=1;
echo \############################### START FOLD 3 LAYER5 \###############################
python train.py --fusion="layer5" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=2;
echo \############################### START FOLD 4 LAYER5 \###############################
python train.py --fusion="layer5" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=3;
echo \############################### START FOLD 5 LAYER5 \###############################
python train.py --fusion="layer5" --dataset=$DATASET --reid=$REID --fuse=$FUSE --fold=4;