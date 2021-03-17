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
#  for reid in 'VtoV' 'TtoT';
#  do
#    python test.py --fusion='unimodal' --dataset=$DATASET --reid=$reid --trained=$reid --fuse='none' --LOO=$LOO ;
#  done
#  for fuse in 'sum' 'cat' 'cat_channel';
#  do
#    for fusion in 'early' 'layer1' 'layer2' 'layer3' 'layer4' 'layer5' ;
#    do
#      python test.py --fusion=$fusion --dataset=$DATASET --reid="BtoB" --trained="BtoB" --fuse=$fuse --LOO=$LOO ;
#    done
#  done

#  for fusion in 'score' 'fc';

#  for weights in '0' '0.05' '0.1' '0.15' '0.2' '0.25' '0.3' '0.35' '0.40' '0.45' '0.5' '0.55' '0.6' '0.65' '0.7' '0.75' '0.8' '0.85' '0.90' '0.95' '1';
#  do
#    python test.py --fusion='score' --dataset=$DATASET --reid="BtoB" --trained="BtoB" --fuse='none' --LOO=$LOO --weight=$weights ;
#  done
#  for i in 'query' 'gallery';
#  do
  for fusion in 'fc_fuse' 'gmu';
  do
    python test.py --fusion=$fusion --dataset=$DATASET --reid="BtoB" --trained="BtoB" --fuse=$fusion --LOO=$LOO ;
  done
#  done
else
    echo "DO $FUSE only"
      for fusion in 'early' 'layer1' 'layer2' 'layer3' 'layer4' 'layer5' ;
    do
      python test.py --fusion=$fusion --dataset=$DATASET --reid="BtoB" --trained="BtoB" --fuse=$FUSE --LOO=$LOO ;
    done
fi



