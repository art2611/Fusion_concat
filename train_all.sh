#!/bin/sh
REID="BtoB"
read -e -p "Enter the dataset name (SYSU/RegDB/TWorld) :" DATASET
echo $DATASET
if [ "$DATASET" =  "regdb" ]
then
  DATASET="RegDB"
elif [ "$DATASET" =  "sysu" ]
then
  DATASET="SYSU"
fi

#read -e -p "Enter the the fuse type (sum/cat/cat_channel/none) :" FUSE
#echo $FUSE


#for i in `seq 0 4`;
#do
#        echo "BEGINING OF THE FIRST TRAINING : Unimodal RGB - Fold = $i"
#        python train.py --fusion="unimodal" --dataset=$DATASET --reid="VtoV" --fuse="none" --fold=$i;
#done
#
#for i in `seq 0 4`;
#do
#        echo "BEGINING OF THE FIRST TRAINING : Unimodal IR - Fold = $i"
#        python train.py --fusion="unimodal" --dataset=$DATASET --reid="TtoT" --fuse="none" --fold=$i;
#done

#for w in 'sum' 'cat' 'cat_channel';
for w in 'sum';
do
  for j in 'layer3' 'layer4' 'layer5';
  do
    for i in `seq 0 4`;
      do
              echo "BEGINING OF THE FIRST TRAINING : $j - Fold = $i fuse = $w"
              python train.py --fusion=$j --dataset=$DATASET --reid="BtoB" --fuse=$w --fold=$i;
      done
  done
done