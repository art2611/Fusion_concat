import numpy as np
from PIL import Image
import pdb
import os
import random
import sys

data_path = '../Datasets/SYSU/'

#PREPARE 5 FOLDS ids files
if False :
    file_path_train = os.path.join(data_path, 'exp/train_id.txt')
    file_path_valid = os.path.join(data_path, 'exp/val_id.txt')

    ###GET VALID AND TRAIN IDS in one list
    with open(file_path_train, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        all_ids = ids

    with open(file_path_valid, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        all_ids.extend(ids)

    training_lists=['' for i in range(5)]
    val_lists = ['' for i in range(5)]


    # Shuffle randomly the list of ids
    random.seed(0)
    all_ids = random.sample(all_ids, len(all_ids))

    for i in range(5):
        # We go over each ids
        for j in range(1,len(all_ids)+1):
            # Validation is only done on 1/5 of the ids, we have 395 ids which lead to folds of 79 ids
            if j >= 79*i + 1 and j <= (79*(i+1)) :
                val_lists[i] = val_lists[i] + str(all_ids[j-1]) + ','
            # Training is always what is not in the previous interval ( 4/5 des ids ici)
            else :
                training_lists[i]= training_lists[i] + str(all_ids[j-1]) + ','
        # Get rid of the lasts "," which are not needed
        val_lists[i] = val_lists[i].rstrip(val_lists[i][-1])
        training_lists[i] = training_lists[i].rstrip(training_lists[i][-1])

    # We create txt files with the validation ids in it, it is enough for validation as we already have for testing
    for k in range(5) :
        f = open(data_path + f"exp/val_id_{k}.txt", "w+")
        f.write(val_lists[k])
        f.close()
        g = open(data_path + f"exp/train_id_{k}.txt", "w+")
        g.write(training_lists[k])
        g.close()


training_lists = [[] for i in range(5)]
# Get the 5 folds in txt file
for k in range(5):
    with open(data_path + f"exp/train_id_{k}.txt", 'r') as file:
        ids = file.read().splitlines()
        ids = ["%04d" % int(all_ids) for all_ids in ids[0].split(',')]
        all_ids = ids
    training_lists[k] = all_ids

rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
ir_cameras = ['cam3', 'cam6']

### GET imgs associated to ids for each folds
files_rgb_train = [[] for i in range(5)]
files_ir_train = [[] for i in range(5)]
pid2label_train = [[] for i in range(5)]

for k in range(5) :
    for id in sorted(training_lists[k]):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_rgb_train[k].extend(new_files)

        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_ir_train[k].extend(new_files)


for k in range(5):
    # relabel
    pid_container = set()
    for img_path in files_ir_train[k]:
        pid = int(img_path[-13:-9])
        pid_container.add(pid)
    pid2label_train[k] = {pid: label for label, pid in enumerate(pid_container)}
    #print(pid2label_train[k])

fix_image_width = 144
fix_image_height = 288


def read_imgs(train_image,k):
    train_img = []
    train_label = []
    for img_path in train_image:
        # img
        img = Image.open(img_path)
        img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
        pix_array = np.array(img)

        train_img.append(pix_array)

        # label
        pid = int(img_path[-13:-9])

        pid = pid2label_train[k][pid]
        train_label.append(pid)

    return np.array(train_img), np.array(train_label)

#Output .npy files for each fold
for k in range(5):

    # rgb imges train
    train_img, train_label = read_imgs(files_rgb_train[k], k)
    np.save(data_path + f'train_rgb_img_{k}.npy', train_img)
    np.save(data_path +  f'train_rgb_label_{k}.npy', train_label)

    # ir imges train
    train_img, train_label = read_imgs(files_ir_train[k], k)
    np.save(data_path +  f'train_ir_img_{k}.npy', train_img)
    np.save(data_path + f'train_ir_label_{k}.npy', train_label)
