import numpy as np
import torch.nn as nn
import torch
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import random
from sklearn.preprocessing import minmax_scale
from PIL import Image
import os
import math
from random import shuffle
import torch.nn.functional as F
# example of a normalization
from numpy import asarray
from sklearn.preprocessing import MinMaxScaler
# define data

#DATA NORMALISATION
# data = np.array([[6., 2.,4.],
# 				[4., 12.,6.]])
# print(math.sqrt(6*6 + 2*2 + 4*4))
# norm = np.linalg.norm(data ,ord=2, axis=1)
# print(data[k][])
# print(np.linalg.norm(data ,ord=2, axis=1))
# print()
# min = np.min(data, axis=1)
# print(min)
# max = np.max(data,axis=1)
# print(data.shape)
# for k in range(data.shape[0]) :
#     for i in range(data.shape[1]):
#         data[k][i] = (data[k][i] - min[k]) / (max[k] - min[k])
# print(data)
a = 0.
while a <=1. :
    print(a)
    a += 0.05
sys.exit()
class GatedBimodal(nn.Module):
    u"""Gated Multimodal Unit neural network - Bimodal use"""
    def __init__(self, dim):
        super(GatedBimodal, self).__init__()

        self.dim = dim
        self.activation = nn.Tanh()
        self.gate_activation = nn.Sigmoid()

        # Learnable weights definition - As describe in the paper
        self.Wz = nn.parameter.Parameter(torch.rand(2*dim, dim))
        self.Wt = nn.parameter.Parameter(torch.rand(1, dim))
        self.Wv = nn.parameter.Parameter(torch.rand(1, dim))
        self.Wz.requires_grad = True
        self.Wt.requires_grad = True
        self.Wv.requires_grad = True

    # x1 and x2 will respectively be RGB input and IR thermal input.
    def forward(self, x1, x2):
        # Prepare the cat tensor for incoming z calcul
        x = torch.cat((x1, x2), 1) # torch.Size([batch size, 2 * dim of input features])
        # Get the batch size
        batch_size = x.shape[0]

        # Get vector of scalar. One scalar for each feature from the batch
        hv = self.activation(torch.mm(self.Wv, torch.transpose(x1, 0, 1))) # torch.Size([1, batch size])
        ht = self.activation(torch.mm(self.Wt, torch.transpose(x2, 0, 1))) # torch.Size([1, batch size])

        # Get the weights for weighted sum fusion of the two modalities
        z = self.gate_activation(torch.mm(x, self.Wz)) # torch.Size([batch size, dim of input features])

        # Prepare the fused feature tensor of size [batch size , dim input feature)
        fused_feat = torch.rand(batch_size, self.dim)

        # For each feature from batch, return the weighted sum
        for k in range(batch_size) :
            fused_feat[k] = z[k][:]*hv[0][k] + (1-z[k][:])*ht[0][k]  # torch.Size([1, dim of input feature])

        # Get the fused features and the matrix of weight, in a way to see which modality contributed the most further
        return fused_feat, z


a = torch.rand((64,500))
b = torch.rand((64,500))

net = GatedBimodal(a.shape[1])
# a = torch.tensor([[1,2],[3,4]])
# b = torch.tensor([[1,2],[3,4]])
# print(a)
# print(b)
# print(a.shape[1])
# print(torch.dot(a[0], b))
# print(a[0]*b[0])
# print(a[1]*b[1])
print(net(a,b))

sys.exit()
w= []
ids = [0,0,0,0, 1,1,1,1,1,1]
ids2 = [2,3]
w.append(ids)
w.append(ids2)
print(w)

sys.exit()
list = []
for k in range(409):
    list.append()
    img_dir_RGB = data_path + '/' + str(k) + '/'
    img_dir_IR = data_path_ir + '/' + str(k) + '/'
    new_files_RGB = sorted([img_dir_RGB + '/' + i for i in os.listdir(img_dir_RGB)])
    new_files_IR = sorted([img_dir_IR + '/' + i for i in os.listdir(img_dir_IR)])
    files_rgb_train.extend(new_files_RGB)
    files_ir_train.extend(new_files_IR)


# define min max scaler

sys.exit()


# print(minmax_scale(np.array([[1,5,12],[5,3,12]]), axis=1))


    # pix_array = np.array(img)
#     train_img.append(pix_array)
#     train_img = np.array(train_img)
# print(train_img.shape)
# for i in range(len(files_rgb_train)):

def read_imgs(train_image,k):
    train_img = []
    train_label = []
    for img_path in train_image:
        # img
        img = Image.open(img_path)
        img = img.resize((144, 288), Image.ANTIALIAS)
        pix_array = np.array(img)

        train_img.append(pix_array)

        # label
        # pid = int(img_path[-13:-9])

        # pid = pid2label_train[k][pid]
        # train_label.append(pid)

    return np.array(train_img)
    # return np.array(train_img), np.array(train_label)
train_img = read_imgs(files_rgb_train, 0)
print(train_img)
w=0
for i in range(44, 56):
    w += 1
    print(i)
    plt.subplot(5,5,w)
    plt.imshow(train_img[i])
plt.show()
sys.exit()

# rgb_tensor = torch.tensor(np.array([[1, 2, 3], [3, 2, 1]]))
# ir_tensor = torch.tensor(np.array([[1, 2, 3], [1, 2, 3]]))
# summed_tensor = rgb_tensor.add(ir_tensor)
# print(f" summed_tensor : {summed_tensor}")


x = np.array([1, 3, 4, 6])
y = np.array([2, 3, 5, 1])
y2 = np.sin(x)
plt.plot(x, y, label="cos(x)")
plt.plot(x, y2, label="sin(x)")

plt.legend()
plt.xlabel("abscisses")
plt.ylabel("ordonnees")
plt.title("Fonction plop")

plt.show()

# affiche la figure a l'ecran
# a = torch.randint(10, (2, 2))
# b = torch.randint(10, (2, 2))
# print(a)
# print(b)
# c = torch.cat((a, b), dim=-1)
#
# print(c)
# print(c.shape)
sys.exit()


def fonction(x):
    return (x + 2)


a = {"bonjour": fonction, "salut": 3}
print(a["bonjour"](2))

a = [[2, 0], [4, 6], [0, 1]]
a = np.array(a)
print(a)

dist = pdist(a, metric='euclidean')
print(dist)
print(a.max(1))
