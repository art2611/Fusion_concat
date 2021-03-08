import torch
import sys
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
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

gamma = 0.7
epochs = 14
optimizer = optim.Adam(net.parameters(), lr=lr)

criterion_id = nn.CrossEntropyLoss().to(device)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
num_epochs= 10
# training

for epochs in range(num_epochs):
    net.train()
    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):
    #     # Labels 1 and 2 could be the same or not. If not : label = 0 If yes : label =  1

        labels = np.array((label1[:] == label2[:])).astype(np.long)
        print(labels)
        labels = torch.from_numpy(labels)

        input1 = Variable(input1.cuda()).float()
        input2 = Variable(input2.cuda()).float()
        labels = Variable(labels.cuda())


        output = net(input1, input2)
        print(output)

        loss = criterion_id(output, labels)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if (batch_idx + 1) % 5 == 0:
            print(f'epochs {epochs + 1} / {num_epochs}, step {batch_idx + 1}/{batch_idx}, loss = {loss.item():.4f}')
    # print(labels)
        #
        # input1 = Variable(input1.cuda())
        # input2 = Variable(input2.cuda())
        # labels = Variable(labels.cuda())





