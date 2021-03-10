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

class GatedBimodal(nn.Module):

    u"""Gated Bimodal neural network.
    Parameters
    ----------
    dim : int
        The dimension of the hidden state.
    activation : :class:`~.bricks.Brick` or None
        The brick to apply as activation. If ``None`` a
        :class:`.Tanh` brick is used.
    gate_activation : :class:`~.bricks.Brick` or None
        The brick to apply as activation for gates. If ``None`` a
        :class:`.Logistic` brick is used.
    Notes
    -----
    See :class:`.Initializable` for initialization parameters.
    """
    def __init__(self, dim, activation=None, gate_activation=None):
        super(GatedBimodal, self).__init__()
        self.dim = dim
        if not activation:
            activation = nn.Tanh()
        if not gate_activation:
            gate_activation = nn.Sigmoid()
        self.activation = activation
        self.gate_activation = gate_activation
        self.W = nn.parameter.Parameter(torch.rand(2*dim, dim))

        # self.initialize()
    # def _allocate(self):
    #     self.W = shared_floatx_nans(
    #         (2 * self.dim, self.dim), name='input_to_gate')
    #     add_role(self.W, WEIGHT)
    #     self.parameters.append(self.W)
    def initialize(self):
        self.weights_init.initialize(self.W, self.rng)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        print(f"x : {x}")
        print(f" x_shape : {x.shape}")

        h = self.activation()(x)
        # h = F.tanh(x)
        print(f" h : {h}")
        print(f" h_shape : {h.shape}")
        # print(x.dot(self.W))
        print(torch.mm(x, self.W))
        # z = self.gate_activation(x.dot(self.W))

        out = torch.rand(510)
        z = self.gate_activation(torch.mm(x, self.W))

        out[k] = z * h[:, :self.dim] + (1 - z) * h[:, self.dim:]

        return out, z
        # return z * h[:, :self.dim] + (1 - z) * h[:, self.dim:], z


a = torch.rand((64,500))
b = torch.rand((64,500))
net = GatedBimodal(a.shape[1])
# a = torch.tensor([[1,2],[3,4]])
# b = torch.tensor([[1,2],[3,4]])
print(a)
print(b)
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
