import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from random import randrange
import random
import torch

import sys

def TrainingData(data_path, dataset, transform, fold):
    if dataset == "SYSU":
        return(SYSUData(data_path, transform=transform, fold = fold))
    elif dataset == "RegDB":
        return(RegDBData(data_path, transform=transform, fold = fold))
    elif dataset == "TWorld":
        return(TWorldDATA(data_path, transform=transform, fold = fold))


class RegDBData(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None, fold = 0):
        # Load training images (path) and labels
        data_dir = '../Datasets/RegDB/'
        #Load color and thermal images + labels
        #Initial images
        train_color_image= np.load( data_dir + f'train_rgb_img_{fold}.npy')
        train_thermal_image = np.load(data_dir + f'train_ir_img_{fold}.npy')

        # Load labels
        # train_color_label = np.load(data_dir + f'train_label_{fold}.npy')
        train_color_label = [int(i/10) for i in range((204-40)*10)]
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

# Get TWorld data for training
class TWorldDATA(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None, fold = 0):
        data_dir = '../Datasets/TWorld/'
        # Load training labels
        labels = np.load(data_dir + f'train_label_{fold}.npy')
        self.train_color_label = labels
        self.train_thermal_label = labels

        # Load training images
        self.train_color_image = np.load(data_dir + f'train_rgb_img_{fold}.npy')
        self.train_thermal_image = np.load(data_dir + f'train_ir_img_{fold}.npy')

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

# Get SYSU data for training
class SYSUData(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None, fold = 0):
        data_dir = '../Datasets/SYSU/'
        # Load training labels
        self.train_color_label = np.load(data_dir + f'train_rgb_label_{fold}.npy')
        self.train_thermal_label = np.load(data_dir + f'train_ir_label_{fold}.npy')

        # Load training images
        self.train_color_image = np.load(data_dir + f'train_rgb_img_{fold}.npy')
        self.train_thermal_image = np.load(data_dir + f'train_ir_img_{fold}.npy')

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

class Features_Data(data.Dataset):
    def __init__(self, dataset, featureIndex1=None, featureIndex2=None, fold = 0):
        # Load training images (path) and labels
        data_dir = f'../Datasets/{dataset}/'
        #Load color and thermal images + labels
        #Initial images
        train_features= np.load( data_dir + f'exp/Features_train_{fold}.npy')

        if dataset == "RegDB" :
            label_features = [int(i/10) for i in range((204-40)*10)]
        elif dataset == "TWorld" :
            label_features = np.load(data_dir + f'train_label_{fold}.npy')

        self.train_features = train_features
        self.train_label_features = label_features

        # Prepare index
        self.cIndex = featureIndex1
        self.tIndex = featureIndex2

    def __getitem__(self, index):
        #Dataset[i] return features from both modal and the corresponding labels
        feat1, target1 = self.train_features[self.cIndex[index]], self.train_label_features[self.cIndex[index]]
        feat2, target2 = self.train_features[self.tIndex[index]], self.train_label_features[self.tIndex[index]]

        return torch.from_numpy(feat1), torch.from_numpy(feat2), target1, target2

    def __len__(self):
        return len(self.train_features)

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

# Call the corresponding dataset function to process data for the validation or the test phase
def process_data(img_dir, mode, dataset, fold=0):
    if dataset=="SYSU":
        img_query, label_query, query_cam, img_gallery, label_gallery, gall_cam= process_sysu(img_dir, mode, fold)
    elif dataset == "RegDB" :
        query_cam, gall_cam = None, None
        img_query, label_query, img_gallery, label_gallery = process_regdb(img_dir, mode, fold)
    elif dataset == "TWorld" :
        query_cam, gall_cam = None, None
        img_query, label_query, img_gallery, label_gallery = process_tworld(img_dir, mode, fold)
    return (img_query, label_query, query_cam, img_gallery, label_gallery, gall_cam)

# Process regDB data for test or validation
def process_tworld(img_dir, mode, fold =  0):

    if mode == "test" :
        input_data_path = img_dir + f'exp/testing.txt'
    if mode == "valid" :
        input_data_path = img_dir + f"exp/val_id_{fold}.txt"
    if mode == "train" :
        input_data_path = img_dir + f"exp/train_id_{fold}.txt"

    ### GET ids in a list
    with open(input_data_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]

    # Get list of list, each sub list containing the images location for one identity
    ids_file_RGB = []
    ids_file_IR = []

    img_dir_init = img_dir

    for id in ids:
        img_dir = img_dir_init +  "/TV_FULL/" + str(id)
        if os.path.isdir(img_dir):
            #Since all images are in a same folder, we get all an id here
            new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
            ids_file_RGB.append(new_files)
        img_dir = img_dir_init +  "/IR_8/" + str(id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
            ids_file_IR.append(new_files)

    img_query = []
    img_gallery = []
    label_query = []
    label_gallery = []
    # Query and gallery are the same since we want to compare query to all gallery image
    for id in range(len(ids)):
        for i in range(len(ids_file_RGB[id])):
            img_gallery.append([ids_file_RGB[id][i], ids_file_IR[id][i]])
            img_query.append([ids_file_RGB[id][i], ids_file_IR[id][i]])
            label_query.append(ids[id])
            label_gallery.append(ids[id])

    return (img_query, np.array(label_query), img_gallery, np.array(label_gallery))


# Process regDB data for test or validation
def process_regdb(img_dir, mode, fold = 0 ):
    if mode == "test" :
        input_visible_data_path = img_dir + f'exp/test_visible_{1}.txt'
        input_thermal_data_path = img_dir + f'exp/test_thermal_{1}.txt'
    if mode == "valid" :
        input_visible_data_path = img_dir + f"exp/val_id_RGB_{fold}.txt"
        input_thermal_data_path = img_dir + f"exp/val_id_IR_{fold}.txt"
    if mode == "train" :
        input_visible_data_path = img_dir + f"exp/train_id_RGB_{fold}.txt"
        input_thermal_data_path = img_dir + f"exp/train_id_IR_{fold}.txt"

    with open(input_visible_data_path) as f:
        data_file_list = open(input_visible_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        ids_file_RGB = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label_visible = [int(s.split(' ')[1]) for s in data_file_list]

    with open(input_thermal_data_path) as f:
        data_file_list = open(input_thermal_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        ids_file_IR = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label_thermal = [int(s.split(' ')[1]) for s in data_file_list]

    ids = np.unique(file_label_visible)

    img_query = []
    img_gallery = []
    label_query = []
    label_gallery = []

    number_images_for_id_k = 10

    # Query and gallery are the same since we want to compare query to all gallery image
    for id in range(len(ids)):
        files_ir = ids_file_IR[id*10:(id+1)*10]
        files_rgb = ids_file_RGB[id*10:(id+1)*10]
        # Here we have 10 images per id for this dataset
        for i in range(10):
            img_gallery.append([files_rgb[i], files_ir[i]])
            img_query.append([files_rgb[i], files_ir[i]])
            label_query.append(id)
            label_gallery.append(id)

    return (img_query, np.array(label_query), img_gallery, np.array(label_gallery))

# Process SYSU data for test or validation
def process_sysu(data_path, method, fold = 0 ):

    # rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
    # ir_cameras = ['cam3', 'cam6']
    fold_or_trial = int(fold)
    if method == "test":
        print("Test set called")
        input_data_path = data_path + f'exp/test_id.txt'
        input_query_gallery_path = data_path + f'exp/query_gallery_test.txt'
        fold_or_trial_total_number=10
    elif method == "valid":
        input_data_path = os.path.join(data_path, f'exp/val_id_{fold}.txt')
        input_query_gallery_path = data_path + f'exp/query_gallery_validation.txt'
        fold_or_trial_total_number=5

    ### GET ids in a list
    with open(input_data_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]
    ### Get the saved random position for query - gallery
    positions_list_RGB = [[] for i in range(fold_or_trial_total_number)]
    positions_list_IR = [[] for i in range(fold_or_trial_total_number)]
    modality = 1
    trial_number = 0
    with open(input_query_gallery_path, 'r') as query_gallery_file:
        for lines in query_gallery_file:
            the_line = lines.strip()
            positions = the_line.splitlines()
            if positions[0] == "modality":
                modality = 2
            elif positions[0] == "fold_or_trial":
                trial_number += 1
                modality = 1
            if positions[0] != "fold_or_trial" and positions[0] != "modality":
                if modality == 1:
                    positions_list_RGB[trial_number].append([int(y) for y in positions[0].split(',')])
                elif modality == 2:
                    positions_list_IR[trial_number].append([int(y) for y in positions[0].split(',')])

    ids_file_RGB = []
    ids_file_IR = []
    ### Get list of list containing images per identity
    for id in ids:
        files_ir, files_rgb = image_list_SYSU(id, data_path)
        ids_file_RGB.append(files_rgb)
        ids_file_IR.append(files_ir)

    img_query = []
    img_gallery = []
    label_query = []
    label_gallery = []
    # Get the wanted query-gallery with corresponding labels
    for id in range(len(ids)):
        files_ir = ids_file_IR[id]
        files_rgb = ids_file_RGB[id]
        # Same for RGB and IR due to preprocessed selection of positions
        number_images_for_id_k = len(positions_list_RGB[fold_or_trial][id])

        for i in range(number_images_for_id_k):
            # Get one images as query
            if i == 0:
                img_query.append([files_rgb[positions_list_RGB[fold_or_trial][id][i]], files_ir[positions_list_IR[fold_or_trial][id][i]]])
                label_query.append(ids[id])
            # Get the remaining as gallery :
            else:
                img_gallery.append([files_rgb[positions_list_RGB[fold_or_trial][id][i]], files_ir[positions_list_IR[fold_or_trial][id][i]]])
                label_gallery.append(ids[id])
    # Just give different cam id to not have problem during SYSU evaluation
    gall_cam = [4 for i in range(len(img_gallery))]
    query_cam = [1 for i in range(len(img_query))]
    return img_query, np.array(label_query), np.array(query_cam), img_gallery, np.array(label_gallery), np.array(gall_cam)

# Get all images concerning one id from the differents cameras in two distinct lists
def image_list_SYSU(id, data_path) :
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



class Prepare_set(data.Dataset):
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