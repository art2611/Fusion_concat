import numpy as np
import torch
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import random
from PIL import Image
import os
import math
#Get ThermalWorld height and width

def Average(lst):
    return sum(lst) / len(lst)

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

def get_height_and_width_variations(files_1, files_2):
    minwidth = 10000
    maxwidth = 0
    minheight = 10000
    maxheight = 0
    total_width = 0
    total_height = 0
    # Get the max lenght or height :
    for image in files_1:
        img = Image.open(image)
        total_width += img.size[0]
        total_height += img.size[1]
        if maxwidth < img.size[0]:
            maxwidth = img.size[0]
        if minwidth > img.size[0]:
            minwidth = img.size[0]
        if maxheight < img.size[1]:
            maxheight = img.size[1]
        if minheight > img.size[1]:
            minheight = img.size[1]
    for image in files_2:
        img = Image.open(image)
        total_width += img.size[0]
        total_height += img.size[1]
        if maxwidth < img.size[0]:
            maxwidth = img.size[0]
        if minwidth > img.size[0]:
            minwidth = img.size[0]
        if maxheight < img.size[1]:
            maxheight = img.size[1]
        if minheight > img.size[1]:
            minheight = img.size[1]
    numb_of_rgb_img = len(files_1)
    numb_of_ir_img = len(files_2)
    sum = numb_of_rgb_img + numb_of_ir_img

    print(f"Number of RGB images : {len(files_1)}")
    print(f"Number of IR images : {len(files_2)}")

    print(f"[maxwidth, minwidth] : [{maxwidth}, {minwidth}]")
    print(f"[maxheight, minheight] : [{maxheight}, {minheight}]")
    print(f"mean_width  : {total_width/sum}")
    print(f"mean_height : {total_height/sum}")
    print(f"Number of total images : {len(files_1)}")

#Function to extract 20% of the data randomly from TWorld
def random_thermalWORLD_training_testing() :
    validation = []
    testing = []
    all = [y for y in range(409)]

    # Get ~20% of data for testing. This is done considering the upcoming 5 folds validation => For folds of same size, 84 for testing is great

    while len(testing) < 84 :
        randuuum = random.randint(0, 408)
        if randuuum not in testing:
            testing.append(randuuum)
    testing.sort()

    for k in all:
        if k not in testing:
            validation.append(k)
    print(f"all : {all}")
    print(f"testing : {testing}")
    print(validation)

    f = open('training.txt', 'a')

    for k in range(len(validation) - 1):
        f.write(str(validation[k]) + ',')
    f.write(str(validation[len(validation) - 1]))
    f.close
    g = open('testing.txt', 'a')
    for k in range(len(testing) - 1):
        g.write(str(testing[k]) + ',')
    g.write(str(testing[len(testing) - 1]))
    g.close

dataset = "nope"

if dataset == "ThermalWorld" :
    data_path = '../Datasets/TWorld/TV_FULL'
    data_path_ir = '../Datasets/TWorld/IR_8'
    files_rgb_train = []
    files_ir_train = []

    Number_id_RGB = []

    i = 0
    for k in range(409):
        img_dir_RGB = data_path + '/' + str(k) + '/'
        img_dir_IR = data_path_ir + '/' + str(k) + '/'
        new_files_RGB = sorted([img_dir_RGB + '/' + i for i in os.listdir(img_dir_RGB)])
        new_files_IR = sorted([img_dir_IR + '/' + i for i in os.listdir(img_dir_IR)])
        if len(Number_id_RGB) < i + 1:
            Number_id_RGB.append(len(new_files_RGB))
        else:
            Number_id_RGB[i] = Number_id_RGB[i] + len(new_files_RGB)
        files_rgb_train.extend(new_files_RGB)
        files_ir_train.extend(new_files_IR)
        i = i +1
    print(f" [Min_image, max_image, AVG_image] : [{min(Number_id_RGB)}, {max(Number_id_RGB)}, {Average(Number_id_RGB)}]")

# Get height and width for sysu dataset
if dataset == "SYSU" :
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

    Number_id_RGB = []
    Number_id_IR = []
    i = 0
    for id in sorted(training_lists):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                if len(Number_id_RGB) < i + 1 :
                    Number_id_RGB.append(len(new_files))
                else :
                    Number_id_RGB[i] = Number_id_RGB[i] + len(new_files)
                files_rgb_train.extend(new_files)

        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                if len(Number_id_IR) < i + 1 :
                    Number_id_IR.append(len(new_files))
                else :
                    Number_id_IR[i] = Number_id_IR[i] + len(new_files)
                files_ir_train.extend(new_files)

        i = i + 1
    distance = []
    for k in range(len(Number_id_RGB)) :
        distance.append(abs(Number_id_RGB[k] - Number_id_IR[k]))

    print(f" [Min_RGB_image, max_RGB_image, AVG_RGB_image] : [{min(Number_id_RGB)}, {max(Number_id_RGB)}, {Average(Number_id_RGB)}]")
    print(f" [Min_ir_image, max_ir_image, AVG_ir_image] : [{min(Number_id_IR)}, {max(Number_id_IR)}, {Average(Number_id_IR)}]")
    print(f" [Min_distance, max_distance, AVG_distance] : [{min(distance)}, {max(distance)}, {Average(distance)}]")

# Get variations in height and width in the selected dataset
if False :
    get_height_and_width_variations(files_rgb_train, files_ir_train)

# Get a training testing random id split
if False :
    random_thermalWORLD_training_testing()


# Generate query gallery random positions for TWorld (first two positions will be used as query and the remaining as gallery)
if False :
    img_dir = '../Datasets/TWorld/'
    input_data_path = img_dir + f'exp/testing.txt'
    trials = 30
    ### GET ids in a list
    with open(input_data_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
    print(len(ids))
    ids_file_RGB = []
    ids_file_IR = []
    img_dir_init = img_dir
    # For all ids :
    for id in ids :
        img_dire = img_dir_init + "/TV_FULL/" + str(id)
        if os.path.isdir(img_dire):
            # Get the list of images for one id
            new_files = sorted([img_dire + '/' + i for i in os.listdir(img_dire)])
            # Get the list of list, containing RGB images per identity
            ids_file_RGB.append(new_files)
        img_dire = img_dir_init + "/IR_8/" + str(id)
        if os.path.isdir(img_dire):
            new_files = sorted([img_dire + '/' + i for i in os.listdir(img_dire)])
            ids_file_IR.append(new_files)
    print(len(ids_file_RGB[1]))
    f = open(img_dir + "exp/" + 'query_gallery_test.txt', 'w')
    query_random_selection = [[] for i in range(10)]
    gallery_random_selection = [[] for i in range(10)]

    for j in range(trials):
        random.seed(j)
        for k in range(len(ids)):
            files_ir = ids_file_IR[k]
            files_rgb = ids_file_RGB[k]
            number_images_for_id_k = len(files_rgb)
            images_position_list = [ i for i in range(number_images_for_id_k)]
            # print(number_images_for_id_k)
            smpl = random.sample(images_position_list, number_images_for_id_k)

            for w in range(number_images_for_id_k - 1) :
                f.write(f"{smpl[w]},")
            f.write(f"{smpl[number_images_for_id_k - 1]}\n")
        if j < trials - 1 :
            f.write(f"fold_or_trial\n")

    # f = open(img_dir + "exp/" + 'query_gallery_validation.txt', 'w')
    # for fold in range(5) :
    #     input_data_path = img_dir + f'exp/val_id_{fold}.txt'
    #
    #     ### GET ids in a list
    #     with open(input_data_path, 'r') as file:
    #         ids = file.read().splitlines()
    #         ids = [int(y) for y in ids[0].split(',')]
    # print(len(ids))

    #     ids_file_RGB = []
    #     ids_file_IR = []
    #     img_dir_init = img_dir
    #     # For all ids :
    #     for id in ids :
    #         img_dire = img_dir_init + "/TV_FULL/" + str(id)
    #         if os.path.isdir(img_dire):
    #             # Get the list of images for one id
    #             new_files = sorted([img_dire + '/' + i for i in os.listdir(img_dire)])
    #             # Get the list of list, containing RGB images per identity
    #             ids_file_RGB.append(new_files)
    #         img_dire = img_dir_init + "/IR_8/" + str(id)
    #         if os.path.isdir(img_dire):
    #             new_files = sorted([img_dire + '/' + i for i in os.listdir(img_dire)])
    #             ids_file_IR.append(new_files)
    #
    #     query_random_selection = [[] for i in range(10)]
    #     gallery_random_selection = [[] for i in range(10)]
    #
    #     random.seed(fold)
    #     for k in range(len(ids)):
    #         files_ir = ids_file_IR[k]
    #         files_rgb = ids_file_RGB[k]
    #         number_images_for_id_k = len(files_rgb)
    #         images_position_list = [ i for i in range(number_images_for_id_k)]
    #
    #         smpl = random.sample(images_position_list, number_images_for_id_k)
    #
    #         for w in range(number_images_for_id_k - 1) :
    #             f.write(f"{smpl[w]},")
    #         f.write(f"{smpl[number_images_for_id_k - 1]}\n")
    #     if fold < 4 :
    #         f.write(f"fold_or_trial\n")

# Generate query gallery random positions for SYSU (first two positions will be used as query and the remaining as gallery)
if False :
    # Test query gallery generation
    img_dir = '../Datasets/SYSU/'
    input_data_path = img_dir + f'exp/test_id.txt'
    trials = 30
    ### GET ids in a list
    with open(input_data_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    ids_file_RGB = []
    ids_file_IR = []
    img_dir_init = img_dir
    # For all ids :
    for id in ids :
        files_ir = 0
        for k in [3,6]:
            img_dire = os.path.join(img_dir_init, f'cam{k}', id)
            if os.path.isdir(img_dire):
                if files_ir == 0:
                    files_ir = sorted([img_dire + '/' + i for i in os.listdir(img_dire)])
                else:
                    files_ir.extend(sorted([img_dire + '/' + i for i in os.listdir(img_dire)]))

        files_rgb = 0
        for k in [1,2,4,5]:
            img_dire = os.path.join(img_dir_init, f'cam{k}', id)
            if os.path.isdir(img_dire) :
                if files_rgb == 0:
                    files_rgb = sorted([img_dire + '/' + i for i in os.listdir(img_dire)])
                else:
                    files_rgb.extend(sorted([img_dire + '/' + i for i in os.listdir(img_dire)]))
        ids_file_RGB.append(files_rgb)
        ids_file_IR.append(files_ir)

    f = open(img_dir + "exp/" + 'query_gallery_test.txt', 'w')
    for i in range(trials):
        random.seed(i)
        for k in range(len(ids)):

            files_rgb = ids_file_RGB[k]
            files_ir = ids_file_IR[k]

            number_images_for_id_k_RGB = len(files_rgb)
            number_images_for_id_k_IR = len(files_ir)

            images_position_list_RGB = [i for i in range(number_images_for_id_k_RGB)]
            images_position_list_IR = [i for i in range(number_images_for_id_k_IR)]

            smpl_rgb = random.sample(images_position_list_RGB, number_images_for_id_k_RGB)
            smpl_ir = random.sample(images_position_list_IR, number_images_for_id_k_IR)

            for w in range(min(number_images_for_id_k_RGB, number_images_for_id_k_IR) - 1):
                f.write(f"{smpl_rgb[w]},")
            f.write(f"{smpl_rgb[min(number_images_for_id_k_RGB, number_images_for_id_k_IR) - 1]}\n")

        f.write(f"modality\n")
        for k in range(len(ids)):

            files_rgb = ids_file_RGB[k]
            files_ir = ids_file_IR[k]

            number_images_for_id_k_RGB = len(files_rgb)
            number_images_for_id_k_IR = len(files_ir)

            images_position_list_RGB = [ i for i in range(number_images_for_id_k_RGB)]
            images_position_list_IR = [ i for i in range(number_images_for_id_k_IR)]

            smpl_rgb = random.sample(images_position_list_RGB, number_images_for_id_k_RGB)
            smpl_ir = random.sample(images_position_list_IR, number_images_for_id_k_IR)

            for w in range(min(number_images_for_id_k_RGB, number_images_for_id_k_IR) - 1):
                f.write(f"{smpl_ir[w]},")
            f.write(f"{smpl_ir[min(number_images_for_id_k_RGB, number_images_for_id_k_IR) - 1]}\n")
        if i < trials :
            f.write(f"fold_or_trial\n")

    # Validation query gallery generation
    img_dir = '../Datasets/SYSU/'
    f = open(img_dir + "exp/" + 'query_gallery_validation.txt', 'w')
    for fold in range(5):
        input_data_path = img_dir + f'exp/val_id_{fold}.txt'

        ### GET ids in a list
        with open(input_data_path, 'r') as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            ids = ["%04d" % x for x in ids]

        ids_file_RGB = []
        ids_file_IR = []
        img_dir_init = img_dir
        # For all ids :
        for id in ids :
            files_ir = 0
            for k in [3,6]:
                img_dire = os.path.join(img_dir_init, f'cam{k}', id)
                if os.path.isdir(img_dire):
                    if files_ir == 0:
                        files_ir = sorted([img_dire + '/' + i for i in os.listdir(img_dire)])
                    else:
                        files_ir.extend(sorted([img_dire + '/' + i for i in os.listdir(img_dire)]))

            files_rgb = 0
            for k in [1,2,4,5]:
                img_dire = os.path.join(img_dir_init, f'cam{k}', id)
                if os.path.isdir(img_dire) :
                    if files_rgb == 0:
                        files_rgb = sorted([img_dire + '/' + i for i in os.listdir(img_dire)])
                    else:
                        files_rgb.extend(sorted([img_dire + '/' + i for i in os.listdir(img_dire)]))
            ids_file_RGB.append(files_rgb)
            ids_file_IR.append(files_ir)


        random.seed(fold)
        for k in range(len(ids)):

            files_rgb = ids_file_RGB[k]
            files_ir = ids_file_IR[k]

            number_images_for_id_k_RGB = len(files_rgb)
            number_images_for_id_k_IR = len(files_ir)

            images_position_list_RGB = [i for i in range(number_images_for_id_k_RGB)]
            images_position_list_IR = [i for i in range(number_images_for_id_k_IR)]

            smpl_rgb = random.sample(images_position_list_RGB, number_images_for_id_k_RGB)
            smpl_ir = random.sample(images_position_list_IR, number_images_for_id_k_IR)

            for w in range(min(number_images_for_id_k_RGB, number_images_for_id_k_IR) - 1):
                f.write(f"{smpl_rgb[w]},")
            f.write(f"{smpl_rgb[min(number_images_for_id_k_RGB, number_images_for_id_k_IR) - 1]}\n")

        f.write(f"modality\n")
        for k in range(len(ids)):

            files_rgb = ids_file_RGB[k]
            files_ir = ids_file_IR[k]

            number_images_for_id_k_RGB = len(files_rgb)
            number_images_for_id_k_IR = len(files_ir)

            images_position_list_RGB = [ i for i in range(number_images_for_id_k_RGB)]
            images_position_list_IR = [ i for i in range(number_images_for_id_k_IR)]

            smpl_rgb = random.sample(images_position_list_RGB, number_images_for_id_k_RGB)
            smpl_ir = random.sample(images_position_list_IR, number_images_for_id_k_IR)

            for w in range(min(number_images_for_id_k_RGB, number_images_for_id_k_IR) - 1):
                f.write(f"{smpl_ir[w]},")
            f.write(f"{smpl_ir[min(number_images_for_id_k_RGB, number_images_for_id_k_IR) - 1]}\n")
        if fold < 4 :
            f.write("fold_or_trial\n")



#Get SYSU Query _ gallery repartition from file for test
if False :
    data_path = '../Datasets/SYSU/'
    input_data_path = data_path + f'exp/test_id.txt'
    input_query_gallery_path = data_path + f'exp/query_gallery.txt'

    ### GET ids in a list
    with open(input_data_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]
    ### Get the saved random position for query - gallery
    positions_list_RGB = [[] for i in range(10)]
    positions_list_IR = [[] for i in range(10)]
    modality = 1
    trial_number = 0
    with open(input_query_gallery_path, 'r') as query_gallery_file:
        for lines in query_gallery_file:
            the_line = lines.strip()
            positions = the_line.splitlines()
            if positions[0] == "modality":
                modality = 2
            elif positions[0] == "trial":
                trial_number += 1
                modality = 1
            if positions[0] != "trial" and positions[0] != "modality":
                if modality == 1:
                    positions_list_RGB[trial_number].append([int(y) for y in positions[0].split(',')])
                elif modality == 2:
                    positions_list_IR[trial_number].append([int(y) for y in positions[0].split(',')])

    ids_file_RGB = []
    ids_file_IR = []
    ### Get list of list containing images per identity
    for id in sorted(ids):
        files_ir, files_rgb = image_list_SYSU(id, data_path)
        ids_file_RGB.append(files_rgb)
        ids_file_IR.append(files_ir)

    fold = 0
    trial = fold
    img_query = []
    img_gallery = []
    label_query = []
    label_gallery = []
    # Get the wanted query-gallery with corresponding labels
    for id in range(len(ids)):
        files_ir = ids_file_IR[id]
        files_rgb = ids_file_RGB[id]
        # Same for RGB and IR due to preprocessed selection of positions
        number_images_for_id_k = len(positions_list_RGB[trial][id])

        for i in range(number_images_for_id_k):
            # Get two images as gallery
            if i < 2:
                img_gallery.append([files_rgb[positions_list_RGB[fold][id][i]], files_ir[positions_list_IR[fold][id][i]]])
                label_gallery.append(ids[id])
            # Get the remaining as query :
            else:
                img_query.append([files_rgb[positions_list_RGB[fold][id][i]], files_ir[positions_list_IR[fold][id][i]]])
                label_query.append(ids[id])

    # Just give different cam id to not have problem during SYSU evaluation
    gall_cam = [4 for i in range(len(img_gallery))]
    query_cam = [1 for i in range(len(img_query))]
    print(ids[0])
    print(img_gallery[0])

#Generate RegDB query/ gallery repartition in file for RegDB test
if True :
    img_dir = '../Datasets/RegDB/'
    input_data_path = img_dir + f'idx/test_visible_{1}.txt'
    trials = 30
    ### GET ids in a list
    with open(input_data_path, 'r') as file:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        ids_file_RGB = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label_visible = [int(s.split(' ')[1]) for s in data_file_list]
    ids = np.unique(file_label_visible)

    f = open(img_dir + "exp/" + 'query_gallery_test.txt', 'w')
    for i in range(trials):
        random.seed(i)
        for k in range(len(ids)):
            files_rgb = ids_file_RGB[k*10:(k+1)*10]
            number_images_for_id_k = len(files_rgb)
            images_position_list = [ i for i in range(number_images_for_id_k)]

            smpl = random.sample(images_position_list, number_images_for_id_k)

            for w in range(number_images_for_id_k - 1) :
                f.write(f"{smpl[w]},")
            f.write(f"{smpl[number_images_for_id_k - 1]}\n")
        if i < trials - 1 :
            f.write(f"fold_or_trial\n")

    # #Generate for validation folds
    # img_dir = '../Datasets/RegDB/'
    # f = open(img_dir + "exp/" + 'query_gallery_validation.txt', 'w')
    # for i in range(5) :
    #
    #     input_data_path = img_dir + f'idx/val_id_RGB_{i}.txt'
    #
    #     ### GET ids in a list
    #     with open(input_data_path, 'r') as file:
    #         data_file_list = open(input_data_path, 'rt').read().splitlines()
    #         # Get full list of image and labels
    #         ids_file_RGB = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
    #         file_label_visible = [int(s.split(' ')[1]) for s in data_file_list]
    #     ids = np.unique(file_label_visible)
    #
    #     random.seed(w)
    #     for k in range(len(ids)):
    #         files_rgb = ids_file_RGB[k*10:(k+1)*10]
    #         number_images_for_id_k = len(files_rgb)
    #         images_position_list = [ i for i in range(number_images_for_id_k)]
    #
    #         smpl = random.sample(images_position_list, number_images_for_id_k)
    #
    #
    #         for w in range(number_images_for_id_k - 1) :
    #             f.write(f"{smpl[w]},")
    #         f.write(f"{smpl[number_images_for_id_k - 1]}\n")
    #     print(i)
    #     if i<4 :
    #         f.write(f"fold_or_trial\n")