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
    def __init__(self, data_dir, trial, transform=None, colorIndex=None, thermalIndex=None, fold = 0):
        # Load training images (path) and labels
        data_dir = '../Datasets/RegDB/'
        #Load color and thermal images + labels
        train_color_image = np.load( data_dir + f'train_rgb_img_{fold}.npy')
        train_thermal_image = np.load(data_dir + f'train_ir_img_{fold}.npy')
        train_color_label = np.load(data_dir + f'train_label_{fold}.npy')
        train_color_label = train_color_label.tolist()
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

class SYSUData_clean(data.Dataset):
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

class SYSUData(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None):
        data_dir = '../Datasets/SYSU/'
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')

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


def process_test_regdb(img_dir, modal='visible', trial = 1, split="paper_based"):

    input_visible_data_path = img_dir + f'idx/test_visible_{trial}.txt'
    input_thermal_data_path = img_dir + f'idx/test_thermal_{trial}.txt'

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

    #If required, return half of the dataset in two slice
    if modal == "VtoV" :
        file_image = file_image_visible
        file_label = file_label_visible
    if modal == "TtoT" :
        file_image = file_image_thermal
        file_label = file_label_thermal
    if modal == "TtoT" or modal == "VtoV" :
        first_image_slice_query = []
        first_label_slice_query = []
        sec_image_slice_gallery = []
        sec_label_slice_gallery = []
        #On regarde pour chaque id
        for k in range(len(np.unique(file_label))):
            appeared=[]
            # On choisit cinq personnes en query aléatoirement, le reste est placé dans la gallery (5 images)
            for i in range(5):
                rand = random.choice(file_image[k*10:k*10+9])
                while rand in appeared:
                    rand = random.choice(file_image[k*10:k*10+9])
                appeared.append(rand)
                first_image_slice_query.append(rand)
                first_label_slice_query.append(file_label[k*10])
            #On regarde la liste d'images de l'id k, on récupère les images n'étant pas dans query (5 images)
            for i in file_image[k*10:k*10+10] :
                if i not in appeared :
                    sec_image_slice_gallery.append(i)
                    sec_label_slice_gallery.append(file_label[k*10])
        return(first_image_slice_query, np.array(first_label_slice_query), sec_image_slice_gallery, np.array(sec_label_slice_gallery))

    if split == "paper_based":
        if modal == "VtoT" :
            return (file_image_visible, np.array(file_label_visible), file_image_thermal,
                np.array(file_label_thermal))
        elif modal == "TtoV":
            return(file_image_thermal, np.array(file_label_thermal), file_image_visible, np.array(file_label_visible))

    elif split=="experience_based" :
        first_image_slice_query = []
        first_label_slice_query = []
        sec_image_slice_gallery = []
        sec_label_slice_gallery = []
        first_image_slice_query_2 = []
        sec_image_slice_gallery_2 = []
        # On regarde pour chaque id
        for k in range(len(np.unique(file_label_visible))):
            appeared = []
            # On choisit cinq personnes en query aléatoirement, le reste est placé dans la gallery (5 images)
            for i in range(5):
                rand = randrange(10)
                while rand in appeared:
                    rand = randrange(10)
                appeared.append(rand)
                if modal == "VtoT" :
                    first_image_slice_query.append(file_image_visible[k * 10 + rand])
                    first_label_slice_query.append(file_label_visible[k * 10])
                elif modal == "TtoV" :
                    first_image_slice_query.append(file_image_thermal[k * 10 + rand])
                    first_label_slice_query.append(file_label_thermal[k * 10])
                elif modal == "BtoB" :
                    first_image_slice_query.append(file_image_visible[k * 10 + rand])
                    first_image_slice_query_2.append(file_image_thermal[k * 10 + rand])
                    first_label_slice_query.append(file_label_visible[k * 10])

            # On regarde la liste d'images de l'id k, on récupère les images n'étant pas dans query (5 images)
            for i in [w for w in range(10)]:
                if i not in appeared:
                    if modal == "VtoT":
                        sec_image_slice_gallery.append(file_image_visible[k * 10 + i])
                        sec_label_slice_gallery.append(file_label_visible[k * 10])
                    elif modal == "TtoV":
                        sec_image_slice_gallery.append(file_image_thermal[k * 10 + i])
                        sec_label_slice_gallery.append(file_label_thermal[k * 10])
                    elif modal =="BtoB" :
                        sec_image_slice_gallery.append(file_image_visible[k * 10 + i])
                        sec_image_slice_gallery_2.append(file_image_thermal[k * 10 + i])
                        sec_label_slice_gallery.append(file_label_visible[k * 10])

        if modal == "BtoB" :
            return (first_image_slice_query, first_image_slice_query_2, np.array(first_label_slice_query), \
                    sec_image_slice_gallery, sec_image_slice_gallery_2, np.array(sec_label_slice_gallery))
        return (first_image_slice_query, np.array(first_label_slice_query), sec_image_slice_gallery,
                np.array(sec_label_slice_gallery))


def process_query_sysu(data_path, method, trial=0, mode='all', relabel=False, reid="VtoT"):
    random.seed(trial)
    print("query")
    if mode == 'all':
        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
        ir_cameras = ['cam3', 'cam6']
    elif mode == 'indoor':
        rgb_cameras = ['cam1', 'cam2']
        ir_cameras = ['cam3', 'cam6']

    if method == "test":
        print("Test set called")
        file_path = os.path.join(data_path, 'exp/test_id.txt')
    elif method == "valid":
        print("Validation set called")
        file_path = os.path.join(data_path, f'exp/val_id_{0}.txt')
        # file_path = os.path.join(data_path, 'exp/val_id.txt')

    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_rgb.extend(new_files)
        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)

    query_img = []
    query_id = []
    query_cam = []
    if reid=="VtoT" :
        files = files_rgb
    elif reid=="TtoV" :
        files = files_ir
    if reid in ["VtoT", "TtoV"]:
        for img_path in files:
            camid, pid = int(img_path[-15]), int(img_path[-13:-9])
            query_img.append(img_path)
            query_id.append(pid)
            query_cam.append(camid)
    # Ajout pour la fusion avec utilisation des deux images :
    if reid == "BtoB":
        # On doit faire attention que l'on n'ai pas un nombre moins grands d'images d'une des modalités
        w = 0
        x = 0
        for k in range(min(len(files_rgb), len(files_ir))):
            pid_rgb = int(files_rgb[x][-13:-9])
            pid_ir = int(files_ir[w][-13:-9])

            if pid_rgb == pid_ir :
                w+=1
                x+=1
                query_img.append([files_rgb[x], files_ir[w]])
                query_id.append(pid_rgb)
                # La cam on doit juste la choisir différente de la cam gallery pour que les calculs de distances soient ok
                query_cam.append(1)
            elif pid_rgb > pid_ir :
                #"pid_rgb > pid_ir "
                #supress img IR : {files_ir[w]}
                w += 1
            elif pid_rgb < pid_ir :
                # pid_rgb < pid_ir
                # supress img : {files_rgb[x]}
                x +=1
    #print(query_img)
    return query_img, np.array(query_id), np.array(query_cam)


def process_gallery_sysu(data_path, method, mode='all', trial=0, relabel=False, reid="VtoT"):
    random.seed(trial)

    if mode == 'all':
        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
        ir_cameras = ['cam3', 'cam6']
    elif mode == 'indoor':
        rgb_cameras = ['cam1', 'cam2']
        ir_cameras = ['cam3', 'cam6']

    if method == "test":
        print("Test set called")
        file_path = os.path.join(data_path, 'exp/test_id.txt')
    elif method == "valid" :
        print("Validation set called")
        file_path = os.path.join(data_path, f'exp/val_id_{0}.txt')

    files_rgb = []
    files_ir = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_rgb.append(random.choice(new_files))
            # else :
            #     print(f'this dir does not exist : {img_dir}')
        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_ir.append(random.choice(new_files))
    gall_img = []
    gall_id = []
    gall_cam = []
    if reid=="VtoT" :
        files = files_ir
    elif reid=="TtoV" :
        files = files_rgb
    if reid in ["VtoT", "TtoV"]:
        for img_path in files:
            camid, pid = int(img_path[-15]), int(img_path[-13:-9])
            gall_img.append(img_path)
            gall_id.append(pid)
            gall_cam.append(camid)
    # Ajout pour la fusion avec utilisation des deux images :
    if reid == "BtoB":
        # On doit faire attention que l'on n'ai pas un nombre moins grands d'images d'une des modalités
        w = 0
        x = 0
        for k in range(min(len(files_rgb), len(files_ir))):
            pid_rgb = int(files_rgb[x][-13:-9])
            pid_ir = int(files_ir[w][-13:-9])
            if pid_rgb == pid_ir :
                w+=1
                x+=1
                gall_img.append([files_rgb[x], files_ir[w]])
                gall_id.append(pid_rgb)
                # La cam on doit juste la choisir différente de la cam gallery pour que les calculs de distances soient ok
                gall_cam.append(4)
            elif pid_rgb > pid_ir :
                #"pid_rgb > pid_ir "
                #supress img IR : {files_ir[w]}
                w += 1
            elif pid_rgb < pid_ir :
                # pid_rgb < pid_ir
                # supress img : {files_rgb[x]}
                x +=1
    return gall_img, np.array(gall_id), np.array(gall_cam)

def process_test_single_sysu(data_path, method, trial=0, mode='all', relabel=False, reid="VtoT"):
    random.seed(trial)
    print("query")
    if mode == 'all':
        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
        ir_cameras = ['cam3', 'cam6']
    elif mode == 'indoor':
        rgb_cameras = ['cam1', 'cam2']
        ir_cameras = ['cam3', 'cam6']

    if method == "test":
        print("Test set called")
        file_path = os.path.join(data_path, 'exp/test_id.txt')
    elif method == "valid":
        print("Validation set called")
        file_path = os.path.join(data_path, 'exp/val_id.txt')

    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]


    files_query_visible = []
    files_gallery_visible = []
    files_query_thermal = []
    files_gallery_thermal = []
    for id in sorted(ids):
        #Selection of 1 img for gallery per cam and per id, the rest as query
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                rand = random.choice(new_files)
                files_gallery_visible.append(rand)
                for w in new_files:
                    if w != rand:
                        files_query_visible.append(w)

        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                rand = random.choice(new_files)
                files_gallery_thermal.append(rand)
                for w in new_files:
                    if w != rand:
                        files_query_thermal.append(w)
    query_img = []
    query_id = []
    query_cam = []
    gall_img = []
    gall_id = []
    gall_cam = []

    if reid == "VtoV":
        files_query = files_query_visible
        files_gallery = files_gallery_visible
    elif reid == "TtoT":
        files_query = files_query_thermal
        files_gallery = files_gallery_thermal

    for img_path in files_query:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)

    for img_path in files_gallery :
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)

    return query_img, np.array(query_id), np.array(query_cam), gall_img, np.array(gall_id), np.array(gall_cam)

def process_BOTH_sysu(data_path, method, fold=0):
    # random.seed(0)

    rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
    ir_cameras = ['cam3', 'cam6']

    if method == "test":
        print("Test set called")
        file_path = os.path.join(data_path, 'exp/test_id.txt')
    elif method == "valid":
        print("Validation set called")
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
            print("One of file is not 0 ")
            temp_gallery_visible = []
            temp_gallery_thermal = []
            temp_query_visible = []
            temp_query_thermal = []

            #Selection of two
            rand_ir = [random.choice(files_ir)]
            rand_ir2 = random.choice(files_ir)
            while rand_ir2 in rand_ir:
                rand_ir2 = random.choice(files_ir)
            rand_ir.append(rand_ir2)
            temp_gallery_thermal = [rand_ir[0], rand_ir[1]]
            for w in files_ir:
                if w not in rand_ir:
                    temp_query_thermal.append(w)

            rand_rgb = [random.choice(files_rgb)]
            rand_rgb2 = random.choice(files_rgb)
            while rand_rgb2 in rand_rgb:
                rand_rgb2 = random.choice(files_rgb)
            rand_rgb.append(rand_rgb2)
            temp_gallery_visible = [rand_rgb[0], rand_rgb[1]]
            for w in files_rgb:
                if w not in rand_rgb:
                    temp_query_thermal.append(w)

            #Get the same number of images for each modality => the minimal available images per id of each modality
            for k in range(min(len(temp_query_visible), len(temp_query_thermal))) :
                files_query_visible.append(temp_query_visible[k])
                files_query_thermal.append(temp_query_thermal[k])
            for k in range(min(len(temp_gallery_visible), len(temp_gallery_thermal))) :
                files_gallery_visible.append(temp_gallery_visible[k])
                files_gallery_thermal.append(temp_query_visible[k])
        else :
            print("One of files is 0 ")
    query_img = []
    query_id = []
    query_cam = []
    gall_img = []
    gall_id = []
    gall_cam = []

    print(len(files_query_visible))
    print(len(files_gallery_visible))
    print(len(files_query_thermal))
    print(len(files_gallery_thermal))
    for k in range(3):
        print(f"visible  : {files_query_visible[k]}")
        print(f"thermal : {files_query_thermal[k]}")
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
        query_img[counter][1] = img_path
        counter += 1
    # print(query_img)
    return query_img, np.array(query_id), np.array(query_cam), gall_img, np.array(gall_id), np.array(gall_cam)

def image_list(id, data_path) :
    files_ir = 0
    for k in [3,6]:
        img_dir = os.path.join(data_path, f'cam{k}', id)
        if os.path.isdir(img_dir):
            print("TRUE")
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