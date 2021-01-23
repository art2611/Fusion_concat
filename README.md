# Fusion_concat

This repository contain the code for a concatenation fusion and a summation fusion.
The fusion is done between two different types of data, RGB and IR thermal.

# Preparation 
First both SYSU-MM01 dataset and RegDB dataset should be downloaded.
Then the following commands should be used to prepare both datasets : 

    python pre_process_sysu_clean.py
    python pre_process_regdb_clean.py

The datasets have to be at this position form this repository : \
../Datasets/SYSU\
../Datasets/RegDB\

# Use 

The command to launch a training are the following : 

    python train.py 

Parameters : \
'--fusion', default='layer1', help='Which layer to fuse (early, layer1, layer2 .., layer5, unimodal)')
parser.add_argument('--fuse', default='cat', help='Fusion type (cat / sum)')
parser.add_argument('--fold', default='0', help='Fold number (0 to 4)')
parser.add_argument('--dataset', default='regdb', help='dataset name (regdb / sysu )')
parser.add_argument('--reid', default='BtoB', help='Type of ReID (BtoB / TtoT / TtoT)')
    
The command to launch a testing is the following : 

    python test.sh 
    

    
# Bash files 
   
You can launch a full training (On 5 folds) by using the next command and a testing (on 5 folds) with the second command  : 

    bash train.sh 
    bash test.sh
    
Then you will have to give the options, which are the same as those described earlier.

