import numpy as np
import torch
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import random
from sklearn.preprocessing import minmax_scale
from PIL import Image
import os


# example of a normalization
from numpy import asarray
from sklearn.preprocessing import MinMaxScaler
# define data

data = np.array([[6., 2.,4.],
				[4., 12.,6.]])

scaler = MinMaxScaler()
scaler.fit(data)
print(scaler.transform(data))
min = np.amin(data, axis=1)
max = np.amax(data,axis=1)
print(data.shape)
for k in range(data.shape[0]) :
    for i in range(data.shape[1]):
        data[k][i] = (data[k][i] - min[k]) / (max[k] - min[k])
print(data)



# define min max scaler


sys.exit()


# print(minmax_scale(np.array([[1,5,12],[5,3,12]]), axis=1))

#Get ThermalWorld height and width
if False :
    data_path = '../Datasets/ThermalWorld/TV_FULL'
    data_path_ir = '../Datasets/ThermalWorld/IR_8'
    files_rgb_train = []
    files_ir_train = []

    for k in range(409):
        img_dir_RGB = data_path + '/' + str(k) + '/'
        img_dir_IR = data_path_ir + '/' + str(k) + '/'
        new_files_RGB = sorted([img_dir_RGB + '/' + i for i in os.listdir(img_dir_RGB)])
        new_files_IR = sorted([img_dir_IR + '/' + i for i in os.listdir(img_dir_IR)])
        files_rgb_train.extend(new_files_RGB)
        files_ir_train.extend(new_files_IR)
    print(f"files_rgb_train : {len(files_rgb_train)}")
    print(f"files_IR_train : {len(files_ir_train)}")
    minwidth = 10000
    maxwidth = 0
    minheight = 10000
    maxheight = 0
    total_width = 0
    total_height = 0
    train_img=[]
    #Get the max lenght or height :
    for image in files_rgb_train :
        img = Image.open(image)
        total_width += img.size[0]
        total_height += img.size[1]
        if maxwidth < img.size[0] :
            maxwidth = img.size[0]
        if minwidth > img.size[0]:
            minwidth = img.size[0]
        if maxheight < img.size[1] :
            maxheight = img.size[1]
        if minheight > img.size[1]:
            minheight = img.size[1]
    print(f"maxlenght : {maxwidth}")
    print(f"minlenght : {minwidth}")
    print(f"maxheight : {maxheight}")
    print(f"minheight : {minheight}")
    print(f"mean_width  : {total_width/8125}")
    print(f"mean_height : {total_height/8125}")
    # pix_array = np.array(img)
#     train_img.append(pix_array)
#     train_img = np.array(train_img)
# print(train_img.shape)
# for i in range(len(files_rgb_train)):


# Get height and width for sysu dataset
if True :
    rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
    ir_cameras = ['cam3', 'cam6']

    data_path = '../Datasets/SYSU/'
    file_path_train = os.path.join(data_path, 'exp/all_id.txt')
    ###GET VALID AND TRAIN IDS in one list
    with open(file_path_train, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        all_ids = ids

    print(len(all_ids))
    training_lists = []
    for j in range(1,len(all_ids)+1):
        training_lists.append("%04d" % all_ids[j - 1])

    files_rgb_train = []
    files_ir_train = []

    for id in sorted(training_lists):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_rgb_train.extend(new_files)

        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_ir_train.extend(new_files)

print(f"Number of RGB images : {len(files_rgb_train)}")
print(f"Number of IR images : {len(files_ir_train)}")

minwidth = 10000
maxwidth = 0
minheight = 10000
maxheight = 0
total_width = 0
total_height = 0
train_img=[]
#Get the max lenght or height :
for image in files_rgb_train :
    img = Image.open(image)
    total_width += img.size[0]
    total_height += img.size[1]
    if maxwidth < img.size[0] :
        maxwidth = img.size[0]
    if minwidth > img.size[0]:
        minwidth = img.size[0]
    if maxheight < img.size[1] :
        maxheight = img.size[1]
    if minheight > img.size[1]:
        minheight = img.size[1]
for image in files_ir_train :
    img = Image.open(image)
    total_width += img.size[0]
    total_height += img.size[1]
    if maxwidth < img.size[0] :
        maxwidth = img.size[0]
    if minwidth > img.size[0]:
        minwidth = img.size[0]
    if maxheight < img.size[1] :
        maxheight = img.size[1]
    if minheight > img.size[1]:
        minheight = img.size[1]
print(f"[maxwidth, minwidth] : [{maxwidth}, {minwidth}]")
print(f"[maxheight, minheight] : [{maxheight}, {minheight}]")

print(f"mean_width  : {total_width/(len(files_rgb_train) + len(files_ir_train))}")
print(f"mean_height : {total_height/(len(files_rgb_train) + len(files_ir_train))}")
sys.exit()


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
