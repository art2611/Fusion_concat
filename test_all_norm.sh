#!/bin/sh
read -e -p "Enter the dataset name (SYSU/RegDB/TWorld) :" DATASET
echo $DATASET


for i in 'fc' 'score';
do
  for j in 'l2norm' 'minmax' 'tanh' 'zscore' 'none';
  do
              echo "testing $j norm for $i fusion on $DATASET dataset"
              python test.py --fusion=$i --dataset=$DATASET --reid="BtoB" --trained="BtoB" --fuse="none" --norm=$j ;

  done
done