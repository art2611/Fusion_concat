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

read -e -p "Enter the the LOO needed (Query / Gallery) :" LOO
echo $LOO

#read -e -p "Enter the the fuse type (sum/cat/cat_channel/none) :" FUSE
#echo $FUSE


#for i in `seq 0 4`;
#do
#        echo "BEGINING OF THE FIRST TRAINING : Unimodal RGB - Fold = $i"
#        python train.py --fusion="unimodal" --dataset=$DATASET --reid="VtoV" --fuse="none" --fold=$i --LOO=$LOO;
#done
#
#for i in `seq 0 4`;
#do
#        echo "BEGINING OF THE FIRST TRAINING : Unimodal IR - Fold = $i"
#        python train.py --fusion="unimodal" --dataset=$DATASET --reid="TtoT" --fuse="none" --fold=$i --LOO=$LOO;
#done


for w in 'sum' 'cat' 'cat_channel';
do
  for j in 'early' 'layer1' 'layer2' 'layer3' 'layer4' 'layer5';
  do
    for i in `seq 0 4`;
      do
              echo "BEGINING OF THE FIRST TRAINING : $j - Fold = $i fuse = $w"
              python train.py --fusion=$j --dataset=$DATASET --reid="BtoB" --fuse=$w --fold=$i --LOO=$LOO;
      done
  done
done