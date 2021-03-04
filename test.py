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
lr = 0.001
checkpoint_path = '../save_model/'
#
parser = argparse.ArgumentParser(description='PyTorch Multi-Modality Training')
parser.add_argument('--fusion', default='layer1', help='Which layer fusion to test (early, layer1, layer2 .., layer5, unimodal, score, fc)')
parser.add_argument('--fuse', default='cat', help='Fusion type (cat / sum)')
parser.add_argument('--dataset', default='RegDB', help='dataset name (RegDB / SYSU )')
parser.add_argument('--reid', default='BtoB', help='Type of ReID (BtoB / TtoT / TtoT)')
parser.add_argument('--trained', default='BtoB', help='Trained model (BtoB / VtoV / TtoT)')
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

writer = SummaryWriter(f"runs/{args.trained}_{args.fusion}_FusionModel_{args.reid}_fusiontype({args.fuse})_test_{args.dataset}_day{d1}_{time.time()}")


# Function to extract gallery features
def extract_gall_feat(gall_loader, ngall, net, modality="VtoV"):
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

            feat_pool, feat_fc, feat = net(input1, input2, fuse=args.fuse, modality = modality)

            # If we want to test cross modal reid with our multi modal models, keep those elifs
            # elif args.reid == "VtoT" or args.reid == "TtoT":
            #     test_mode = 2
            #     feat_pool, feat_fc = net(input2, input2, modal=test_mode, fuse = args.fuse)
            # elif args.reid == "TtoV" or args.reid == "VtoV":
            #     test_mode = 1
            #     feat_pool, feat_fc = net(input1, input1, modal=test_mode, fuse = args.fuse)

            gall_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            gall_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            gall_final_fc[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    return gall_feat_pool, gall_feat_fc, gall_final_fc

#Function to extract query image features
def extract_query_feat(query_loader, nquery, net, modality="VtoV"):
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0

    query_feat_pool = np.zeros((nquery, 512))
    query_feat_fc = np.zeros((nquery, 512))
    query_final_fc = np.zeros((nquery, 512))

    with torch.no_grad():
        for batch_idx, (input1, input2, label) in enumerate(query_loader):
            batch_num = input1.size(0)

            input1 = Variable(input1.cuda())
            input2 = Variable(input2.cuda())

            feat_pool, feat_fc, feat = net(input1, input2, fuse=args.fuse , modality=modality)
            # If we want to test cross modal reid with our multi modal models, keep those elifs
            # elif args.reid == "VtoT" or args.reid == "TtoT":
            #     test_mode = 2
            #     feat_pool, feat_fc = net(input2, input2, modal=test_mode, fuse = args.fuse)
            # elif args.reid == "TtoV" or args.reid == "VtoV":
            #     test_mode = 1
            #     feat_pool, feat_fc = net(input1, input1, modal=test_mode, fuse = args.fuse)

            # print(feat_pool.shape)
            # print(feat_fc.shape)
            query_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            query_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            query_final_fc[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()

            ptr = ptr + batch_num
        print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    return query_feat_pool, query_feat_fc, query_final_fc

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

mAP_list = []
mINP_list = []
end = time.time()
Fusion_layer = {"early": 0,"layer1":1, "layer2":2, "layer3":3, "layer4":4, "layer5":5, "unimodal":0, "score":0, "fc":0}
# New global model
if args.dataset == "RegDB" or args.dataset == "TWorld":
    nclass = 164
    if args.dataset == "TWorld" :
        nclass = 260
    data_path = f'../Datasets/{args.dataset}/'
    net = []
    net2 = [[],[],[],[],[]]
    # Since we are supposed to have 5 models, this loop get an average result
    for k in range(5):
        suffix = f'{args.dataset}_{args.reid}_fuseType({args.fuse})_{args.fusion}person_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_fold_{k}'
        print('==> Resuming from checkpoint..')
        model_path = checkpoint_path + suffix + '_best.t'

        if args.fusion == "score" or args.fusion=="fc" :
            suffix = f'{args.dataset}_VtoV_fuseType(none)_unimodalperson_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_fold_{k}'
            suffix2 = f'{args.dataset}_TtoT_fuseType(none)_unimodalperson_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_fold_{k}'
            model_path = checkpoint_path + suffix + '_best.t'
            model_path2 = checkpoint_path + suffix2 + '_best.t'
            print(f"model path2 : {model_path2}")

        print(f"model path : {model_path}")
        if os.path.isfile(model_path):
            print('==> loading checkpoint')
            checkpoint = torch.load(model_path)

            net.append(Global_network(nclass, fusion_layer=Fusion_layer[args.fusion]).to(device))
            net[k].load_state_dict(checkpoint['net'])
            print(f"Fold {k} loaded")
            if args.fusion == "score" or args.fusion=="fc":
                if os.path.isfile(model_path2) :
                    print('==> loading checkpoint 2')
                    checkpoint2 = torch.load(model_path2)
                    net2[k] = Global_network(nclass, fusion_layer=Fusion_layer[args.fusion]).to(device)
                    net2[k].load_state_dict(checkpoint2['net'])
                    print(f"Fold {k} loaded")
        else:
            print(f"Fold {k} doesn't exist")
            print(f"==> Model ({model_path}) can't be loaded")

        # loaded_folds = len(net)
        loaded_folds = 5

    # for test_fold in range(loaded_folds):
        test_fold = k
        #Prepare query and gallery

        query_img, query_label, query_cam, gall_img, gall_label, gall_cam = \
            process_data(data_path, "test", args.dataset, test_fold)

        # Gallery and query set
        gallset = Prepare_set(gall_img, gall_label, transform=transform_test, img_size=(img_w, img_h))
        queryset = Prepare_set(query_img, query_label, transform=transform_test, img_size=(img_w, img_h))

        # Validation data loader
        gall_loader = torch.utils.data.DataLoader(gallset, batch_size=test_batch_size, shuffle=False,
                                                  num_workers=workers)
        query_loader = torch.utils.data.DataLoader(queryset, batch_size=test_batch_size, shuffle=False,
                                                   num_workers=workers)
        nquery = len(query_label)
        ngall = len(gall_label)

        print('Data Loading Time:\t {:.3f}'.format(time.time() - end))
        if args.fusion=="score" or args.fusion=="fc":
            args.reid = "VtoV"
        query_feat_pool, query_feat_fc, query_final_fc = extract_query_feat(query_loader, nquery = nquery, net = net[test_fold], modality = args.reid)
        gall_feat_pool,  gall_feat_fc, gall_final_fc = extract_gall_feat(gall_loader, ngall = ngall, net = net[test_fold], modality = args.reid)

        # pool5 feature
        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))

        cmc_pool, mAP_pool, mINP_pool = eval_regdb(-distmat_pool,query_label , gall_label)

        # fc feature
        distmat = np.matmul( query_feat_fc, np.transpose(gall_feat_fc))

        if args.fusion == "score" or args.fusion=="fc":
            # Extraction for the IR images with the model trained on IR modality
            query_feat_pool2, query_feat_fc2, query_final_fc2 = extract_query_feat(query_loader, nquery=nquery, net=net2[test_fold], modality = "TtoT")
            gall_feat_pool2, gall_feat_fc2, gall_final_fc2 = extract_gall_feat(gall_loader, ngall=ngall, net=net2[test_fold], modality = "TtoT")

            if args.fusion == "score" :
                # Proceed to 2nd matching and aggregate matching matrix

                # # print(query_final_fc[0])
                query_final_fc = l2_norm(query_final_fc)
                query_final_fc2 = l2_norm(query_final_fc2)
                gall_final_fc = l2_norm(gall_final_fc)
                gall_final_fc2 = l2_norm(gall_final_fc2)
                distmat = np.matmul(query_final_fc, np.transpose(gall_final_fc))
                distmat2 = np.matmul(query_final_fc2, np.transpose(gall_final_fc2))
                distmat = l2_norm(distmat)
                distmat2 = l2_norm(distmat2)
                distmat = (distmat + 0.25*distmat2)/2
            else :
                # Proceed to a simple feature aggregation, features incoming from the two distinct unimodal trained models (RGB and IR )
                #First do a norm :
                query_final_fc = tanh_norm(query_final_fc)
                query_final_fc2 = tanh_norm(query_final_fc2)
                gall_final_fc = tanh_norm(gall_final_fc)
                gall_final_fc2 = tanh_norm(gall_final_fc2)
                # print(query_final_fc[0])
                # # print(query_final_fc[0])
                #then aggregate all
                query_feat_fc = (query_final_fc + 0.5*query_final_fc2) / 2
                # print(query_feat_fc)
                gall_feat_fc = (gall_final_fc + 0.5*gall_final_fc2) / 2

                distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

        cmc, mAP, mINP = eval_regdb(-distmat,query_label ,gall_label)

        if test_fold == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
            all_mINP_pool = mINP_pool
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
            all_mINP_pool = all_mINP_pool + mINP_pool

        print(f'Test fold: {test_fold}')
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'POOL: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))
        mAP_list.append(mAP)
        mINP_list.append(mINP)

if args.dataset == 'SYSU':
    nclass = 316
    data_path = '../Datasets/SYSU/'
    net = []
    net2 = [[],[],[],[],[]]

    # Since we are supposed to have 5 models, this loop get an average result
    for k in range(5):
        suffix = f'{args.dataset}_{args.reid}_fuseType({args.fuse})_{args.fusion}person_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_fold_{k}'

        print('==> Resuming from checkpoint..')
        model_path = checkpoint_path + suffix + '_best.t'


        if args.fusion=="score" or args.fusion=="fc":
            suffix = f'{args.dataset}_VtoV_fuseType(none)_unimodalperson_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_fold_{k}'
            suffix2 = f'{args.dataset}_TtoT_fuseType(none)_unimodalperson_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_fold_{k}'
            model_path = checkpoint_path + suffix + '_best.t'
            model_path2 = checkpoint_path + suffix2 + '_best.t'
            print(f"model path 2 : {model_path2}")
        print(f"model path : {model_path}")
        if os.path.isfile(model_path):
            print('==> loading checkpoint')

            checkpoint = torch.load(model_path)
            net.append(Global_network(nclass, fusion_layer=Fusion_layer[args.fusion]).to(device))

            # Append the found model in the network list
            net[k].load_state_dict(checkpoint['net'])
            print(f"Fold {k} loaded")
            if args.fusion == "score" or args.fusion=="fc":
                if os.path.isfile(model_path2) :
                    print('==> loading checkpoint 2')
                    checkpoint2 = torch.load(model_path2)
                    net2[k] = (Global_network(nclass, fusion_layer=Fusion_layer[args.fusion]).to(device))
                    net2[k].load_state_dict(checkpoint2['net'])
                    print(f"Fold {k} loaded")
        else :
            print(f"Fold {k} doesn't exist")
            print(f"==> Model ({model_path}) can't be loaded")

        # Get the data and display all
        query_img, query_label, query_cam, gall_img, gall_label, gall_cam = process_sysu(data_path, "test", fold=0)

        gallset = Prepare_set(gall_img, gall_label, transform=transform_test, img_size=(img_w, img_h))
        queryset = Prepare_set(query_img, query_label, transform=transform_test, img_size=(img_w, img_h))

        gall_loader = torch.utils.data.DataLoader(gallset, batch_size=test_batch_size, shuffle=False,
                                                  num_workers=workers)
        query_loader = torch.utils.data.DataLoader(queryset, batch_size=test_batch_size, shuffle=False,
                                                   num_workers=workers)
        nquery = len(query_label)
        ngall = len(gall_label)

        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
        print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
        print("  ------------------------------")
        print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

        # loaded_folds = len(net)
        loaded_folds = 5

        # for test_fold in range(loaded_folds):
        test_fold = k
        # Extract normalized distances with the differents trained networks (from fold 0 to 4)

        # Extraction for RGB if score or fc fusion
        if args.fusion=="score" or args.fusion=="fc":
            args.reid = "VtoV"
        query_feat_pool, query_feat_fc, query_final_fc = extract_query_feat(query_loader, nquery=nquery, net=net[test_fold], modality = args.reid)
        gall_feat_pool, gall_feat_fc, gall_final_fc = extract_gall_feat(gall_loader,ngall = ngall, net = net[test_fold], modality = args.reid)

        # pool5 feature
        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        cmc_pool, mAP_pool, mINP_pool = eval_sysu(-distmat_pool, query_label, gall_label, query_cam, gall_cam)

        # EXtraction for IR if score or fc fusion
        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
        # print(f"distmat : {distmat}")
        # print(f"gallfeat  : {gall_feat_fc}")
        if args.fusion == "score" or args.fusion=="fc":
            # Extraction for the IR images with the model trained on IR modality
            query_feat_pool2, query_feat_fc2, query_final_fc2 = extract_query_feat(query_loader, nquery=nquery, net=net2[test_fold], modality = "TtoT")
            gall_feat_pool2, gall_feat_fc2, gall_final_fc2 = extract_gall_feat(gall_loader, ngall=ngall, net=net2[test_fold], modality = "TtoT")

            if args.fusion == "score" :
                # Proceed to 2nd matching and aggregate matching matrix
                query_final_fc = l2_norm(query_final_fc)
                # print(query_final_fc[0])
                query_final_fc2 = l2_norm(query_final_fc2)
                gall_final_fc = l2_norm(gall_final_fc)
                gall_final_fc2 = l2_norm(gall_final_fc2)
                distmat = np.matmul(query_final_fc, np.transpose(gall_final_fc))
                distmat2 = np.matmul(query_final_fc2, np.transpose(gall_final_fc2))
                distmat = l2_norm(distmat)
                distmat2 = l2_norm(distmat2)
                distmat = (distmat + distmat2)/2
            else :
                # Proceed to a simple feature aggregation, features incoming from the two distinct unimodal trained models (RGB and IR )
                #First do a minmax norm :
                # print(query_final_fc[0])
                # query_final_fc = minmax_norm(query_final_fc)
                # # print(query_final_fc[0])
                # query_final_fc2 = minmax_norm(query_final_fc2)
                # gall_final_fc = minmax_norm(gall_final_fc)
                # gall_final_fc2 = minmax_norm(gall_final_fc2)

                #then aggregate all
                query_feat_fc = (query_feat_fc + query_feat_fc2) / 2
                # print(query_feat_fc)
                gall_feat_fc = (gall_feat_fc + gall_feat_fc2) / 2

                distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)

        if test_fold == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
            all_mINP_pool = mINP_pool
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
            all_mINP_pool = all_mINP_pool + mINP_pool

        print(f'Test fold: {test_fold}')
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('POOL: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))
        mAP_list.append(mAP)
        mINP_list.append(mINP)

#Standard Deviation :
standard_deviation_mAP = np.std(mAP_list)
standard_deviation_mINP = np.std(mINP_list)
# Means
cmc = all_cmc / loaded_folds
mAP = all_mAP / loaded_folds
mINP = all_mINP / loaded_folds

cmc_pool = all_cmc_pool / loaded_folds
mAP_pool = all_mAP_pool / loaded_folds
mINP_pool = all_mINP_pool / loaded_folds
print('All Average:')
print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%} | stdmAP: {:.2%} | stdmINP {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP, standard_deviation_mAP, standard_deviation_mINP))

f = open('results.txt','a')
if args.fusion == "unimodal" :
    f.write(f"{args.dataset}_{args.fusion}_{args.fuse}_{args.reid}\n")
else :
    f.write(f"{args.dataset}_{args.fusion}_{args.fuse} : (1 on RGB - 1 on IR )\n")
f.write('FC: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%} | stdmAP: {:.2%} | stdmINP {:.2%}\n\n'.format(
        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP, standard_deviation_mAP, standard_deviation_mINP))
f.close()

# print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
# cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))

for k in range(len(cmc)):
    writer.add_scalar('cmc curve', cmc[k]*100, k + 1)