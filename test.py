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
from evaluation import *
from torchvision import transforms
import torch.utils.data
from multiprocessing import freeze_support
from tensorboardX import SummaryWriter
import argparse
from datetime import date
from Normalization_func import *

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
parser.add_argument('--fusion', default='score', help='Which layer fusion to test (early, layer1, layer2 .., layer5, unimodal, score, fc)')
parser.add_argument('--fuse', default='none', help='Fusion type (cat / sum)')
parser.add_argument('--dataset', default='RegDB', help='dataset name (RegDB / SYSU )')
parser.add_argument('--reid', default='BtoB', help='Type of ReID (BtoB / TtoT / TtoT)')
parser.add_argument('--trained', default='BtoB', help='Trained model (BtoB / VtoV / TtoT)')
parser.add_argument('--norm', default='l2norm', help='Normalization for feat or score (l2norm, zscore, minmax, tanh)')
parser.add_argument('--LOO', default='query', help='Leave one out (query / gallery)')
parser.add_argument('--weight', default='0.5', help='Weighted summation for modalities')

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

# writer = SummaryWriter(f"runs/{args.trained}_{args.fusion}_FusionModel_{args.reid}_fusiontype({args.fuse})_test_{args.dataset}_day{d1}_{time.time()}")


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


# Init Var
mAP_list = []
mINP_list = []
trials = 30
folds = 5
mAP_mINP_per_trial = {"mAP" : [0 for i in range(trials)], "mINP" : [0 for i in range(trials)]}
mAP_mINP_per_model = {"mAP" : [0 for i in range(folds)], "mINP" : [0 for i in range(folds)]}
end = time.time()
Fusion_layer = {"early": 0,"layer1":1, "layer2":2, "layer3":3, "layer4":4, "layer5":5, "fc_fuse": 5, "gmu":5, "unimodal":0, "score":0, "fc":0}
nclass = {"RegDB" : 164, "SYSU" : 316, "TWorld" : 260}
Need_two_trained_unimodals = {"early": False,"layer1":False, "layer2":False, \
                              "layer3":False, "layer4":False, "layer5":False, \
                              "unimodal":False, "fc_fuse":False, "gmu" : False, "score" : True, "fc" : True}




if args.dataset == "SYSU" or args.LOO == "gallery":
    trials = 30
else :
    trials = 1

data_path = f'../Datasets/{args.dataset}/'
net = []
net2 = [[] for i in range(folds)]
# Since we are supposed to have 5 models (5 fold validation), this loop get an average result
for fold in range(folds):
    if args.LOO == "query":
        suffix = f'{args.dataset}_{args.reid}_fuseType({args.fuse})_{args.fusion}person_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_fold_{fold}'
    else:
        suffix = f'{args.dataset}_{args.reid}_fuseType({args.fuse})_{args.fusion}person_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_fold_{fold}_LOO_{args.LOO}'
    print('==> Resuming from checkpoint..')
    model_path = checkpoint_path + suffix + '_best.t'

    if Need_two_trained_unimodals[args.fusion] :
        if args.LOO == "query":
            suffix = f'{args.dataset}_VtoV_fuseType(none)_unimodalperson_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_fold_{fold}'
            suffix2 = f'{args.dataset}_TtoT_fuseType(none)_unimodalperson_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_fold_{fold}'
        else:
            suffix = f'{args.dataset}_VtoV_fuseType(none)_unimodalperson_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_fold_{fold}_LOO_{args.LOO}'
            suffix2 = f'{args.dataset}_TtoT_fuseType(none)_unimodalperson_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_fold_{fold}_LOO_{args.LOO}'

        model_path = checkpoint_path + suffix + '_best.t'
        model_path2 = checkpoint_path + suffix2 + '_best.t'
        print(f"model path2 : {model_path2}")

    print(f"model path : {model_path}")
    if os.path.isfile(model_path):
        print('==> loading checkpoint')
        checkpoint = torch.load(model_path)

        net.append(Global_network(nclass[args.dataset], fusion_layer=Fusion_layer[args.fusion]).to(device))
        net[fold].load_state_dict(checkpoint['net'])
        print(f"Fold {fold} loaded")
        if Need_two_trained_unimodals[args.fusion]:
            if os.path.isfile(model_path2) :
                print('==> loading checkpoint 2')
                checkpoint2 = torch.load(model_path2)
                net2[fold] = Global_network(nclass[args.dataset], fusion_layer=Fusion_layer[args.fusion]).to(device)
                net2[fold].load_state_dict(checkpoint2['net'])
                print(f"Fold {fold} loaded")
    else:
        print(f"Fold {fold} doesn't exist")
        print(f"==> Model ({model_path}) can't be loaded")

    for trial in range(trials):

        #Prepare query and gallery
        query_img, query_label, query_cam, gall_img, gall_label, gall_cam = process_data(data_path, "test", args.dataset, args.LOO, trial)

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
        if Need_two_trained_unimodals[args.fusion] :
            #In this case, extraction for the RGB images with the model trained on RGB modality first
            args.reid = "VtoV"
        query_feat_pool, query_feat_fc, query_final_fc = extract_query_feat(query_loader, nquery = nquery, net = net[fold], modality = args.reid)
        gall_feat_pool,  gall_feat_fc, gall_final_fc = extract_gall_feat(gall_loader, ngall = ngall, net = net[fold], modality = args.reid)

        # pool5 feature
        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        # fc feature
        distmat = np.matmul( query_feat_fc, np.transpose(gall_feat_fc))

        if Need_two_trained_unimodals[args.fusion] :
            # In this case, extraction for the IR images with the model trained on IR modality
            query_feat_pool2, query_feat_fc2, query_final_fc2 = extract_query_feat(query_loader, nquery=nquery, net=net2[fold], modality = "TtoT")
            gall_feat_pool2, gall_feat_fc2, gall_final_fc2 = extract_gall_feat(gall_loader, ngall=ngall, net=net2[fold], modality = "TtoT")

            if args.fusion == "score" :
                # Proceed to 2nd matching and aggregate matching matrix
                # print(query_final_fc[0])

                query_final_fc, gall_final_fc = Normalize_func(query_final_fc, gall_final_fc, args.norm)
                query_final_fc2, gall_final_fc2 = Normalize_func(query_final_fc2, gall_final_fc2, args.norm)

                distmat = np.matmul(query_final_fc, np.transpose(gall_final_fc))
                distmat2 = np.matmul(query_final_fc2, np.transpose(gall_final_fc2))

                # distmat, distmat2 = Normalize_func(distmat, distmat2, args.norm)
                print(f"Weight value : {args.weight}")
                distmat = (float(args.weight)*distmat + (1-float(args.weight)) * distmat2)
            elif args.fusion == "fc":
                # Proceed to a simple feature aggregation, features incoming from the two distinct unimodal trained models (RGB and IR )
                #First do a norm :
                query_final_fc, gall_final_fc = Normalize_func(query_final_fc, gall_final_fc, args.norm)
                query_final_fc2, gall_final_fc2 = Normalize_func(query_final_fc2, gall_final_fc2, args.norm)
                #then aggregate all features
                query_feat_fc = (query_final_fc + query_final_fc2) / 2
                gall_feat_fc = (gall_final_fc + gall_final_fc2) / 2

                distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
        if args.dataset == "SYSU" :
            cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
            cmc_pool, mAP_pool, mINP_pool = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        else :
            cmc, mAP, mINP = evaluation(-distmat, query_label, gall_label, LOO = args.LOO)
            cmc_pool, mAP_pool, mINP_pool = evaluation(-distmat_pool, query_label, gall_label, LOO = args.LOO)

        if trial == 0 and fold == 0 :
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

        mAP_mINP_per_trial["mAP"][trial] += mAP
        mAP_mINP_per_trial["mINP"][trial] += mINP
        mAP_mINP_per_model["mAP"][fold] += mAP
        mAP_mINP_per_model["mINP"][fold] += mINP

        print(f'Test fold: {fold} - Test trial : {trial}')
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'POOL: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))


#Standard Deviation :
if args.dataset == "SYSU" or args.LOO == "gallery":
    standard_deviation_mAP_model = np.std([mAP_mINP_per_model["mAP"][k] / trials for k in range(folds)])
    standard_deviation_mINP_model = np.std([mAP_mINP_per_model["mINP"][k] / trials for k in range(folds)])
    standard_deviation_mAP_trial = np.std([mAP_mINP_per_trial["mAP"][k] / folds for k in range(trials)])
    standard_deviation_mINP_trial = np.std([mAP_mINP_per_trial["mINP"][k] / folds for k in range(trials)])
else :
    standard_deviation_mAP_model = np.std(mAP_mINP_per_model["mAP"])
    standard_deviation_mINP_model = np.std(mAP_mINP_per_model["mINP"])
    standard_deviation_mAP_trial = 0
    standard_deviation_mINP_trial = 0

# Means
cmc = all_cmc / (folds * trials)
mAP = all_mAP / (folds * trials)
mINP = all_mINP / (folds * trials)

cmc_pool = all_cmc_pool / (folds * trials)
mAP_pool = all_mAP_pool / (folds * trials)
mINP_pool = all_mINP_pool / (folds * trials)
print('All Average:')
print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%} | stdmAP: {:.2%} | stdmINP {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP, standard_deviation_mAP_model, standard_deviation_mINP_model))


if os.path.isfile("results.txt") :
    f = open('results.txt','a')
else :
    f = open('results.txt','w+')
    f.write(' , Rank-1, Rank-5, mAP, mINP, stdmAP, stdmINP\n')

if args.fusion == "score" or args.fusion == "fc" :
    data_info = f"{args.dataset}_{args.fusion}_{args.fuse}_{args.reid}_{args.norm}_{args.LOO}_weight({args.weight})"
else :
    data_info = f"{args.dataset}_{args.fusion}_{args.fuse}_{args.reid}_{args.LOO}"

f.write(f'  {data_info}, {cmc[0]:.2%}, {cmc[4]:.2%}, {mAP:.2%}±{standard_deviation_mAP_model:.2%},\
    {mINP:.2%}±{standard_deviation_mINP_model:.2%}, std_mAP_trial{standard_deviation_mAP_trial:.2%}, std_mINP_trial{standard_deviation_mINP_trial:.2%}\n')
f.close()
#
# # print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
# # cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))
#
# for k in range(len(cmc)):
#     writer.add_scalar('cmc curve', cmc[k]*100, k + 1)