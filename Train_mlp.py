import torch
import sys
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import time
from data_loader import *
from torchvision import transforms
from utils import IdentityFeatureSampler, AverageMeter, adjust_learning_rate
from loss import BatchHardTripLoss
from tensorboardX import SummaryWriter
from model import Global_network,  MLP
from evaluation import *
import argparse
from datetime import date

parser = argparse.ArgumentParser(description='PyTorch Multi-Modality Training')
parser.add_argument('--fusion', default='unimodal', help='Which layer to fuse (early, layer1, layer2 .., layer5, unimodal)')
parser.add_argument('--fuse', default='none', help='Fusion type (cat / cat_channel / sum)')
parser.add_argument('--fold', default='0', help='Fold number (0 to 4)')
parser.add_argument('--dataset', default='RegDB', help='dataset name (RegDB / SYSU )')
parser.add_argument('--reid', default='VtoV', help='Type of ReID (BtoB / VtoV / TtoT)')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_batch_size = 64
batch_num_identities = 8 # 8 different identities in a batch
num_of_same_id_in_batch = 4 # Number of same identity in a batch
loader_batch = batch_num_identities * num_of_same_id_in_batch
workers = 4
lr = 0.1

data_path = f'../Datasets/{args.dataset}/'

trainset = Features_Data(args.dataset, fold = args.fold)

feature_pos, _ = GenIdx(trainset.train_label_features, trainset.train_label_features)
feature_size = len(trainset.train_features[0])
print(feature_size)
net = MLP().to(device)

print('==> Preparing Data Loader...')
# identity sampler - Give iteratively index from a randomized list of color index and thermal index
sampler = IdentityFeatureSampler(trainset.train_label_features, feature_pos, num_of_same_id_in_batch, batch_num_identities, args.dataset)

trainset.cIndex = sampler.index1  # color index
trainset.tIndex = sampler.index2  # thermal index
# print(epoch)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=loader_batch, \
                                          sampler=sampler, num_workers=workers, drop_last=True)
# print(len(trainloader))

# training
net.train()
for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):
#     # Labels 1 and 2 could be the same or not. If not : label = 0 If yes : label =  1

    labels = np.array((label1[:] == label2[:])).astype(np.int32)
    labels = torch.from_numpy(labels)

    input1 = input1.float()
    input2 = input2.float()
    input1 = Variable(input1.cuda())
    input2 = Variable(input2.cuda())
    labels = Variable(labels.cuda())
    print(input1)
    print(input2)
    output = net(input1, input2)
    print(output)

# print(labels)
    #
    # input1 = Variable(input1.cuda())
    # input2 = Variable(input2.cuda())
    # labels = Variable(labels.cuda())

