#!/bin/sh
read -e -p "Enter the dataset name (SYSU/RegDB/TWorld) :" DATASET
echo $DATASET
if [ "$DATASET" =  "regdb" ]
then
  DATASET="RegDB"
elif [ "$DATASET" =  "sysu" ]
then
  DATASET="SYSU"
fi

read -e -p "Enter the the LOO needed (Query / Gallery) :" LOO
echo $LOO

read -e -p "Enter the fuse type (sum/cat/cat_channel/none / ALL) of the trained model :" FUSE
if [ "$FUSE" =  "ALL" ]
then
  echo "DO ALL"
  # In this case do test of sum cat and cat channel at all position
  for reid in 'VtoV' 'TtoT';
  do
    python test.py --fusion='unimodal' --dataset=$DATASET --reid=$reid --trained=$reid --fuse='none' --LOO=$LOO ;
  done
    for fuse in 'cat';
  do
    for fusion in 'early' 'layer1' 'layer2' 'layer3' 'layer4' 'layer5' ;
    do
      python test.py --fusion=$fusion --dataset=$DATASET --reid="BtoB" --trained="BtoB" --fuse=$fuse --LOO=$LOO ;
    done
  done
else
    echo "DO $FUSE only"
      for fusion in 'early' 'layer1' 'layer2' 'layer3' 'layer4' 'layer5' ;
    do
      python test.py --fusion=$fusion --dataset=$DATASET --reid="BtoB" --trained="BtoB" --fuse=$FUSE --LOO=$LOO ;
    done
fi



