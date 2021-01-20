import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from random import randrange
import random

class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        data_dir = '../Datasets/RegDB/'
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'
        #Load color and thermal images + labels
        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        #Get real and thermal images with good shape in a list
        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)

        train_color_image = np.array(train_color_image)
        train_thermal_image = np.array(train_thermal_image)

        # Init color images / labels
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        # Init themal images / labels
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        self.transform = transform

        # Prepare index
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        #Dataset[i] return images from both modal and the corresponding label
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

class RegDBData_clean(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None, fold = 0):
        # Load training images (path) and labels
        data_dir = '../Datasets/RegDB/'
        #Load color and thermal images + labels
        train_color_image = np.load( data_dir + f'train_rgb_img_{fold}.npy')
        train_thermal_image = np.load(data_dir + f'train_ir_img_{fold}.npy')
        train_color_label = np.load(data_dir + f'train_label_{fold}.npy')
        train_color_label = train_color_label.tolist()
        #same labels for both images
        train_thermal_label = train_color_label

        # Init color images / labels
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        # Init themal images / labels
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        self.transform = transform

        # Prepare index
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        #Dataset[i] return images from both modal and the corresponding label
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

class SYSUData(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None, fold = 0):
        data_dir = '../Datasets/SYSU/'
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + f'train_rgb_img_{fold}.npy')
        self.train_color_label = np.load(data_dir + f'train_rgb_label_{fold}.npy')

        train_thermal_image = np.load(data_dir + f'train_ir_img_{fold}.npy')
        self.train_thermal_label = np.load(data_dir + f'train_ir_label_{fold}.npy')

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label

# generate the idx of each person identity for instance, identity 10 have the index 100 to 109

def GenIdx(train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(train_color_label) if v == unique_label_color[i]]
        color_pos.append(tmp_pos)

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k, v in enumerate(train_thermal_label) if v == unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)

    return color_pos, thermal_pos

def process_BOTH(img_dir, mode, dataset, fold=0):
    if dataset=="sysu":
        img_query, label_query, query_cam, img_gallery, label_gallery, gall_cam= process_BOTH_sysu(img_dir, mode, fold)
    elif dataset == "regdb" :
        query_cam, gall_cam = None, None
        img_query, label_query, img_gallery, label_gallery = process_BOTH_regdb(img_dir, mode, fold)
    return (img_query, label_query, query_cam, img_gallery, label_gallery, gall_cam)

def process_BOTH_regdb(img_dir, mode, fold):

    if mode == "test" :
        input_visible_data_path = img_dir + f'idx/test_visible_{1}.txt'
        input_thermal_data_path = img_dir + f'idx/test_thermal_{1}.txt'
    if mode == "valid" :
        input_visible_data_path = img_dir + f"idx/val_id_RGB_{fold}.txt"
        input_thermal_data_path = img_dir + f"idx/val_id_IR_{fold}.txt"

    with open(input_visible_data_path) as f:
        data_file_list = open(input_visible_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image_visible = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label_visible = [int(s.split(' ')[1]) for s in data_file_list]

    with open(input_thermal_data_path) as f:
        data_file_list = open(input_thermal_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image_thermal = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label_thermal = [int(s.split(' ')[1]) for s in data_file_list]

    img_query = []
    label_query = []
    img_gallery = []
    label_gallery = []

    # On regarde pour chaque id
    for k in range(len(np.unique(file_label_visible))):
        appeared = []

        for i in range(2):
            rand = randrange(10)
            while rand in appeared:
                rand = randrange(10)
            appeared.append(rand)
            # On récupère des images appairées
            img_gallery.append([file_image_visible[k * 10 + rand], file_image_thermal[k * 10 + rand]])
            # On récupère les labels associés (Les mêmes dans file_label_visible/thermal)
            label_gallery.append(file_label_visible[k * 10 + rand])

        for i in range(10):
            if i not in appeared :
                #On récupère les images appairées restantes
                img_query.append([file_image_visible[k * 10 + i], file_image_thermal[k * 10 + i]])

                label_query.append(file_label_visible[k*10 + i])

    return (img_query, np.array(label_query), img_gallery, np.array(label_gallery))

def process_BOTH_sysu(data_path, method, fold):
    # random.seed(0)

    rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
    ir_cameras = ['cam3', 'cam6']

    if method == "test":
        print("Test set called")
        file_path = os.path.join(data_path, 'exp/test_id.txt')
    elif method == "valid":
        file_path = os.path.join(data_path, f'exp/val_id_{fold}.txt')


    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    files_query_visible = []
    files_gallery_visible = []
    files_query_thermal = []
    files_gallery_thermal = []
    minimum = 0
    temp_query_visible = []
    temp_query_thermal = []
    for id in sorted(ids):
        #Instead of selecting 1 img per cam, we want the same amount of img for both modalities
        # So we select randomly 2 img (no matter which cam) per id and per modality, the rest as query but with a pair number

        files_ir, files_rgb = image_list(id, data_path)
        if files_ir != 0 and files_rgb != 0 :
            temp_gallery_visible = []
            temp_gallery_thermal = []
            temp_query_visible = []
            temp_query_thermal = []
            #Selection of two IR images
            rand_ir = [random.choice(files_ir)]
            rand_ir2 = random.choice(files_ir)
            while rand_ir2 in rand_ir:
                rand_ir2 = random.choice(files_ir)
            rand_ir.append(rand_ir2)
            temp_gallery_thermal = [rand_ir[0], rand_ir[1]]
            #Get all the other IR img in a temporary list
            for w in files_ir:
                if w not in rand_ir:
                    temp_query_thermal.append(w)
            #Selection of two RGB images
            rand_rgb = [random.choice(files_rgb)]
            rand_rgb2 = random.choice(files_rgb)
            while rand_rgb2 in rand_rgb:
                rand_rgb2 = random.choice(files_rgb)
            rand_rgb.append(rand_rgb2)
            temp_gallery_visible = [rand_rgb[0], rand_rgb[1]]
            # Get all the other RGB img in a temporary list
            for w in files_rgb:
                if w not in rand_rgb:
                    temp_query_visible.append(w)

            #Get the same number of images for each modality => the minimal available images per id of each modality
            for k in range(min(len(temp_query_visible), len(temp_query_thermal))) :
                files_query_visible.append(temp_query_visible[k])
                files_query_thermal.append(temp_query_thermal[k])
            for k in range(min(len(temp_gallery_visible), len(temp_gallery_thermal))) :
                files_gallery_visible.append(temp_gallery_visible[k])
                files_gallery_thermal.append(temp_query_visible[k])

    query_img = []
    query_id = []
    query_cam = []
    gall_img = []
    gall_id = []
    gall_cam = []

    # print(len(files_query_visible))
    # print(len(files_gallery_visible))
    # print(len(files_query_thermal))
    # print(len(files_gallery_thermal))
    # for k in range(10):
    #     print(f"visible  : {files_gallery_visible[k]}")
    #     print(f"thermal : {files_gallery_thermal[k]}")

    #Finally get the img, the corresponding ids. The cam doesn't matter.
    for img_path in files_query_visible:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append([img_path,img_path])
        query_id.append(pid)
        query_cam.append(1)
    counter = 0
    for img_path in files_query_thermal :
        query_img[counter][1] = img_path
        counter += 1

    for img_path in files_gallery_visible :
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append([img_path,img_path])
        gall_id.append(pid)
        gall_cam.append(4)
    counter = 0
    for img_path in files_gallery_thermal :
        gall_img[counter][1] = img_path
        counter += 1
    # print(query_img)
    return query_img, np.array(query_id), np.array(query_cam), gall_img, np.array(gall_id), np.array(gall_cam)

def image_list(id, data_path) :
    files_ir = 0
    for k in [3,6]:
        img_dir = os.path.join(data_path, f'cam{k}', id)
        if os.path.isdir(img_dir):
            if files_ir == 0:
                files_ir = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
            else:
                files_ir.extend(sorted([img_dir + '/' + i for i in os.listdir(img_dir)]))

    files_rgb = 0
    for k in [1,2,4,5]:
        img_dir = os.path.join(data_path, f'cam{k}', id)
        if os.path.isdir(img_dir) :
            if files_rgb == 0:
                files_rgb = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
            else:
                files_rgb.extend(sorted([img_dir + '/' + i for i in os.listdir(img_dir)]))

    return(files_ir, files_rgb)

class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)

class TestData_both(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image1 = []
        test_image2 = []

        for i in range(len(test_img_file)):
            img1 = Image.open(test_img_file[i][0])
            img2 = Image.open(test_img_file[i][1])
            img1 = img1.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            img2 = img2.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array1 = np.array(img1)
            pix_array2 = np.array(img2)
            test_image1.append(pix_array1)
            test_image2.append(pix_array2)

        test_image1 = np.array(test_image1)
        test_image2 = np.array(test_image2)

        self.test_image1 = test_image1
        self.test_image2 = test_image2
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, img2, target1 = self.test_image1[index], self.test_image2[index], self.test_label[index]
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2, target1

    def __len__(self):
        #Should be the same len for both image 1 and image 2
        return len(self.test_image1)

# Print some of the images :
# print(trainset.train_color_image.shape)
# w=0
# for i in range(0, 250, 10):
#     w += 1
#     print(i)
#     plt.subplot(5,5,w)
#     plt.imshow(trainset.train_color_image[i])
# plt.show()

# testing set
# query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
# gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')