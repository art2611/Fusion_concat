import numpy as np
from PIL import Image
import os
import sys

data_path = '../Datasets/RegDB/'

file_path_train_visible =  os.path.join(data_path, 'idx/train_visible_1.txt')
file_path_train_thermal =  os.path.join(data_path, 'idx/train_thermal_1.txt')


# train_thermal_list = data_dir + 'idx/train_thermal_1.txt'

###GET all initial training ids (50% of the dataset)
with open(file_path_train_visible) as f:
    data_file_list = open(file_path_train_visible, 'rt').read().splitlines()
    # Get full list of image
    raw_line_visible = [s for s in data_file_list]
    file_image_visible = [data_path + '/' + s.split(' ')[0] for s in data_file_list]

with open(file_path_train_thermal) as f:
    data_file_list = open(file_path_train_thermal, 'rt').read().splitlines()
    # Get full list of image and labels
    raw_line_thermal= [s for s in data_file_list]
    file_image_thermal = [data_path + '/' + s.split(' ')[0] for s in data_file_list]
    # Labels are the same for RGB and IR
    file_label = [int(s.split(' ')[1]) for s in data_file_list]



# Remove the last 10 images and labels (We want to have 205 ids instead of 206 cause 205%5=41 and we want 5 folds)
for k in range(10) :
    file_image_visible.pop()
    file_image_thermal.pop()
    file_label.pop()

all_ids = file_label

#PREPARE 5 FOLDS ids
training_lists= [[] for i in range(5)]

for i in range(5):
    # Creation of files with image name and target
    f = open(data_path + f"exp/val_id_RGB_{i}.txt", "w+")
    g = open(data_path + f"exp/val_id_IR_{i}.txt", "w+")
    h = open(data_path + f"exp/train_id_RGB_{i}.txt", "w+")
    l = open(data_path + f"exp/train_id_IR_{i}.txt", "w+")
    for j in range(1,len(all_ids)+1):
        if j >= 41*10*i + 1 and j <= (41*10*(i+1)) :
            f.write(raw_line_visible[j-1] + "\n")
            g.write(raw_line_thermal[j-1] + "\n")
        else :
            # Training images preparation
            h.write(raw_line_visible[j-1] + "\n")
            l.write(raw_line_visible[j-1] + "\n")
            training_lists[i].append(all_ids[j-1])
    f.close()
    g.close()
    h.close()
    l.close()
sys.exit()

### GET imgs associated to ids
files_rgb_train = [[] for i in range(5)]
files_ir_train = [[] for i in range(5)]

for k in range(5) :
    for j in np.unique(training_lists[k]) :
        for i in range(10):
            files_rgb_train[k].append(file_image_visible[j*10 + i])
            files_ir_train[k].append(file_image_thermal[j*10 + i])


# for k in range(5) :
#     print(files_rgb_train[k])
#     print(training_lists[k])
#     print(len(files_ir_train[k]))
#     print(len(files_rgb_val[k]))
#     print(len(files_ir_val[k]))

fix_image_width = 144
fix_image_height = 288
def read_imgs(train_image, k):
    train_img = []
    for img_path in train_image:
        # img
        img = Image.open(img_path)
        img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
        pix_array = np.array(img)

        train_img.append(pix_array)

    return np.array(train_img)


#Output .npy files for each folds

for k in range(5):
    # rgb imges train
    train_img = read_imgs(files_rgb_train[k], k)
    # print(train_img[0])
    np.save(data_path + f'train_rgb_img_{k}.npy', train_img)
    np.save(data_path +  f'train_label_{k}.npy', np.array(training_lists[k]))

    # ir imges train
    train_img  = read_imgs(files_ir_train[k], k)
    np.save(data_path +  f'train_ir_img_{k}.npy', train_img)