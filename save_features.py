import os
import sys
import torch
import torch.utils.data
from torch.autograd import Variable
import time
from data_loader import *
import numpy as np
from model import Global_network
import math
from evaluation import eval_regdb, eval_sysu
from torchvision import transforms
import torch.utils.data
from multiprocessing import freeze_support
from tensorboardX import SummaryWriter
import argparse
from datetime import date


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

pool_dim = 2048

# Init variables :
img_w = 144
img_h = 288
test_batch_size = 64
batch_num_identities = 8  # 8 different identities in a batch
num_of_same_id_in_batch = 4  # Number of same identity in a batch
workers = 4
lr = 0.1
checkpoint_path = '../save_model/'
#
parser = argparse.ArgumentParser(description='PyTorch Multi-Modality Training')
parser.add_argument('--dataset', default='RegDB', help='dataset name (RegDB / SYSU )')
args = parser.parse_args()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_h, img_w)),
    transforms.ToTensor(),
    normalize,
])
today = date.today()
# dd/mm/YY
d1 = today.strftime("%d/%m/%Y")

# Function to extract gallery features
def extract_feat(gall_loader, ngall, net, modality="VtoV"):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0

    gall_feat_pool = np.zeros((ngall, 512))
    gall_feat_fc = np.zeros((ngall, 512))
    gall_final_fc = np.zeros((ngall, 512))
    with torch.no_grad():
        for batch_idx, (input1, input2, label) in enumerate(gall_loader):
            batch_num = input1.size(0)
            input1 = Variable(input1.cuda())
            input2 = Variable(input2.cuda())

            feat_pool, feat_fc, feat = net(input1, input2, fuse="none", modality = modality)

            gall_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            gall_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            gall_final_fc[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    return gall_feat_pool, gall_feat_fc, gall_final_fc

def minmax_norm(data):
    min = np.amin(data, axis=1)
    max = np.amax(data, axis=1)

    for k in range(data.shape[0]):
        for i in range(data.shape[1]):
            data[k][i] = (data[k][i] - min[k]) / (max[k] - min[k])
    return(data)

def Z_mean(data):
    std = np.std(data, axis=1)
    mean = np.mean(data, axis=1)

    for k in range(data.shape[0]):
        for i in range(data.shape[1]):
            data[k][i] = (data[k][i] - mean[k])/ std[k]
    return(data)

def tanh_norm(data):
    std = np.std(data, axis=1)
    mean = np.mean(data, axis=1)

    for k in range(data.shape[0]):
        for i in range(data.shape[1]):
            data[k][i] = 0.5*math.tanh(0.01*(data[k][i] - mean[k])/ std[k])
    return(data)

def l2_norm(data):

    norm_l2 = np.linalg.norm(data, ord=2, axis=1)
    for k in range(data.shape[0]):
        for i in range(data.shape[1]):
            data[k][i] = data[k][i] / norm_l2[k]
    return(data)

def write_features(file, feat_matrix):
    mat_row, mat_columns = feat_matrix.shape[0], feat_matrix.shape[1]
    for i in range(mat_row) :
        for j in range(mat_columns):
            if j == mat_columns-1 :
                file.write(f'{feat_matrix[i][j]},')
            else :
                file.write(f'{feat_matrix[i][j]}\n')

# Init Var
folds = 5

Fusion_layer = {"early": 0,"layer1":1, "layer2":2, "layer3":3, "layer4":4, "layer5":5, "unimodal":0, "score":0, "fc":0}
nclass = {"RegDB" : 164, "SYSU" : 316, "TWorld" : 260}

data_dir = f'../Datasets/{args.dataset}/'
net = []
net2 = [[] for i in range(folds)]
f = open(f'Features_validation_{args.dataset}.txt','w+')
g = open(f'Features_training_{args.dataset}.txt','w+')
# For the 5 folds :
for fold in range(folds):

    suffix = f'{args.dataset}_VtoV_fuseType(none)_unimodalperson_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_fold_{fold}'
    suffix2 = f'{args.dataset}_TtoT_fuseType(none)_unimodalperson_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_fold_{fold}'
    model_path = checkpoint_path + suffix + '_best.t'
    model_path2 = checkpoint_path + suffix2 + '_best.t'
    print(f"model path : {model_path}")
    print(f"model path2 : {model_path2}")

    #Get RGB unimodal model
    if os.path.isfile(model_path):
        print('==> loading checkpoint')
        checkpoint = torch.load(model_path)

        net.append(Global_network(nclass[args.dataset], fusion_layer=Fusion_layer["unimodal"]).to(device))
        net[fold].load_state_dict(checkpoint['net'])
        print(f"Fold {fold} loaded RGB model")
    else:
        print(f"Fold {fold} doesn't exist")
        print(f"==> Model ({model_path}) can't be loaded")

    #Get IR unimodal model
    if os.path.isfile(model_path2) :
        print('==> loading checkpoint 2')
        checkpoint2 = torch.load(model_path2)
        net2[fold] = Global_network(nclass[args.dataset], fusion_layer=Fusion_layer["unimodal"]).to(device)
        net2[fold].load_state_dict(checkpoint2['net'])
        print(f"Fold {fold} loaded IR model")
    else:
        print(f"Fold {fold} doesn't exist")
        print(f"==> Model ({model_path}) can't be loaded")


    #Get features from each folds
    validation_img, validation_label, _, _, _, _ = process_data(data_dir, "test", args.dataset, fold)

    # Get the data used for training and validation
    validation_set = Prepare_set(validation_img, validation_label, transform=transform_test, img_size=(img_w, img_h))
    # validation_data_set = Prepare_set(query_img, query_label, transform=transform_test, img_size=(img_w, img_h))

    # Validation data loader
    data_loader= torch.utils.data.DataLoader(validation_set, batch_size=test_batch_size, shuffle=False, num_workers=workers)

    nimages = len(validation_label)

    # Extraction for the RGB images with the model trained on RGB modality
    _ , _ , RGB_feature_matrix = extract_feat(data_loader, nimages, net = net[fold], modality = "VtoV")

    # Extraction for the IR images with the model trained on IR modality
    _ , _ , IR_feature_matrix = extract_feat(data_loader, nimages, net=net2[fold], modality = "TtoT")
    print(f"IR feature matrix shape : {IR_feature_matrix.shape}")
    feature_matrix = np.concatenate((RGB_feature_matrix, IR_feature_matrix), axis = 0)
    np.save(f"../Datasets/{args.dataset}/exp/Features_validation_{fold}.npy", feature_matrix)

    if True:
        feature_matrix = np.load(f"../Datasets/{args.dataset}/exp/Features_validation_{0}.npy")

        # print(f"loaded matrix feature IR : {feature_matrix[0][0]}")
        # print(f"loaded matrix feature IR : {feature_matrix[int(feature_matrix.shape[0]/2)][0]}")

    # write_features(f, RGB_feature_matrix)
    # f.write('modality')
    # write_features(f, IR_feature_matrix)
    # f.write('fold')

    # Load training images
    train_color_image = np.load(data_dir + f'train_rgb_img_{fold}.npy')
    train_thermal_image = np.load(data_dir + f'train_ir_img_{fold}.npy')
    training_image = []
    if args.dataset == "TWorld" :
        training_label= np.load(data_dir + f'train_label_{fold}.npy')
    # elif args.dataset == "SYSU" :
    #     train_color_label = np.load(data_dir + f'train_rgb_label_{fold}.npy')
    #     train_thermal_label = np.load(data_dir + f'train_ir_label_{fold}.npy')
    elif args.dataset == "RegDB" :
        print("TRUE")
        training_label = [int(i / 10) for i in range((204 - 40) * 10)]

    for k in range(len(train_color_image)):
        training_image.append([train_color_image[k], train_thermal_image[k]])

    training_set = Prepare_set(training_image, training_label, transform=transform_test, img_size=(img_w, img_h))

    # Training data loader
    data_loader= torch.utils.data.DataLoader(training_set, batch_size=test_batch_size, shuffle=False, num_workers=workers)

    n_images = len(training_label)

    # Extraction for the RGB images with the model trained on RGB modality
    RGB_feature_matrix = extract_feat(data_loader, n_images, net = net[fold], modality = "VtoV")

    # Extraction for the IR images with the model trained on IR modality
    IR_feature_matrix = extract_feat(data_loader, n_images, net=net2[fold], modality = "TtoT")

    feature_matrix = np.concatenate((RGB_feature_matrix, IR_feature_matrix), axis = 0)
    np.save(f"../Datasets/{args.dataset}/exp/Features_train_{fold}.npy", feature_matrix)

#     write_features(g, RGB_feature_matrix)
#     f.write('modality')
#     write_features(g, IR_feature_matrix)
#     f.write('fold')
#
# f.close()
# g.close()



if False :
    feature_matrix = np.load(f"../Dataset/{args.dataset}/ext/Features_validation_{0}.npy")

    print(f"loaded matrix feature  : {feature_matrix[0]}" )
    print(f"loaded matrix feature  : {feature_matrix[feature_matrix.shape[0]/2]}" )
    # Get the features in a list from file :
    # fold = 0
    # features_RGB = [[] for i in range(5)]
    # feature_IR = [[] for i in range(5)]
    # with open(f'Features_validation_{args.dataset}.txt', 'r') as Validation_features_file:
    #     for lines in Validation_features_file:
    #         the_line = lines.strip()
    #         positions = the_line.splitlines()
    #         if positions[0] == "modality":
    #             modality = 2
    #         elif positions[0] == "fold":
    #             fold += 1
    #             modality = 1
    #         if positions[0] != "fold" and positions[0] != "modality":
    #             if modality == 1:
    #                 features_RGB[fold].append([int(y) for y in positions[0].split(',')])
    #             elif modality == 2:
    #                 feature_IR[fold].append([int(y) for y in positions[0].split(',')])

#
# # print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
# # cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))
#
# for k in range(len(cmc)):
#     writer.add_scalar('cmc curve', cmc[k]*100, k + 1)