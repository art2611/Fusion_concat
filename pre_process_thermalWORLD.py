import numpy as np
from PIL import Image
import os
import sys

data_path = '../Datasets/ThermalWorld/'

file_path_train =  os.path.join(data_path, 'training.txt')

### GET ids in a list
with open(file_path_train, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    all_ids = ids

#PREPARE 5 FOLDS ids
training_lists=[[],[],[],[],[]]
val_lists = ['','','','','']

# i is the number of folds
for i in range(5):
    # j = on parcourt toutes les identités
    for j in range(1,len(all_ids)+1):
        # Validation is only done on 1/5 of the ids, we have 395 ids which lead to folds of 79 ids
        if j >= 65*i + 1 and j <= (65*(i+1)) :
            if j == 65*(i+1) :
                val_lists[i] = val_lists[i] + str(all_ids[j-1])
            else :
                val_lists[i] = val_lists[i] + str(all_ids[j-1]) + ','
        # Training is always what is not in the previous interval ( 4/5 des ids ici)
        else :
            training_lists[i].append(all_ids[j-1])

for k in range(5) :
    f = open(data_path + f"val_id_{k}.txt", "w+")
    f.write(val_lists[k])
    f.close()


file_path_train_visible = '../Datasets/ThermalWorld/TV_FULL'
file_path_train_thermal = '../Datasets/ThermalWorld/IR_8'

### GET imgs associated to ids for each folds
files_rgb_train = [[], [], [], [], []]
files_ir_train = [[], [], [], [], []]
pid2label_train = [[], [], [], [], []]

for k in range(5) :
    relabel = 0
    for id in sorted(training_lists[k]):
        img_dir = os.path.join(data_path, "TV_FULL", str(id))
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
            for w in range(len(new_files)):
                pid2label_train[k].append(relabel)
            files_rgb_train[k].extend(new_files)
        relabel += 1
        img_dir = os.path.join(data_path, "IR_8", str(id))
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
            files_ir_train[k].extend(new_files)

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

    train_label = pid2label_train[k]

    return np.array(train_img), np.array(train_label)


#Output .npy files for each fold
for k in range(5):
    # rgb images train
    train_img, train_label = read_imgs(files_rgb_train[k], k)
    np.save(data_path + f'train_rgb_img_{k}.npy', train_img)

    # ir images train
    train_img, train_label = read_imgs(files_ir_train[k], k)
    np.save(data_path +  f'train_ir_img_{k}.npy', train_img)

    #Get paired labels
    np.save(data_path +  f'train_label_{k}.npy', train_label)