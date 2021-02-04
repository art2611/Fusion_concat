import os
import sys
import torch
import torch.utils.data
from torch.autograd import Variable
import time
from data_loader import *
import numpy as np
from model_layer1 import Network_layer1
from model_layer2 import Network_layer2
from model_layer3 import Network_layer3
from model_layer4 import Network_layer4
from model_layer5 import Network_layer5
from model_unimodal import Network_unimodal
from model_early import Network_early
from evaluation import eval_regdb, eval_sysu
from torchvision import transforms
import torch.utils.data
from multiprocessing import freeze_support
from tensorboardX import SummaryWriter
import argparse
from datetime import date


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# net = Network(class_num=nclass).to(device)

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
parser.add_argument('--fusion', default='layer1', help='Which layer to fuse (early, layer1, layer2 .., layer5, unimodal)')
parser.add_argument('--fuse', default='cat', help='Fusion type (cat / sum)')
parser.add_argument('--fold', default='0', help='Fold number (0 to 4)')
parser.add_argument('--dataset', default='regdb', help='dataset name (regdb / sysu )')
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
def extract_gall_feat(gall_loader, ngall, net, modality="visible"):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0

    gall_feat_pool = np.zeros((ngall, 512))
    gall_feat_fc = np.zeros((ngall, 512))

    with torch.no_grad():
        for batch_idx, (input1, input2, label) in enumerate(gall_loader):
            batch_num = input1.size(0)
            input1 = Variable(input1.cuda())
            input2 = Variable(input2.cuda())

            if args.reid == "BtoB":
                if args.reid == "unimodal" and args.reid=="TtoT":
                    input1 = input2
                if args.reid == "unimodal" and args.reid=="VtOV":
                    input1 = input1
                test_mode=0
                feat_pool, feat_fc = net(input1, input2, modal=test_mode, fuse=args.fuse, modality = modality)
            # If we want to test cross modal reid with our multi modal models, keep those elifs
            elif args.reid == "VtoT" or args.reid == "TtoT":
                test_mode = 2
                feat_pool, feat_fc = net(input2, input2, modal=test_mode, fuse = args.fuse)
            elif args.reid == "TtoV" or args.reid == "VtoV":
                test_mode = 1
                feat_pool, feat_fc = net(input1, input1, modal=test_mode, fuse = args.fuse)

            gall_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            gall_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    return gall_feat_pool, gall_feat_fc

#Function to extract query image features
def extract_query_feat(query_loader, nquery, net, modality="visible"):
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0

    query_feat_pool = np.zeros((nquery, 512))
    query_feat_fc = np.zeros((nquery, 512))

    with torch.no_grad():
        for batch_idx, (input1, input2, label) in enumerate(query_loader):
            batch_num = input1.size(0)

            input1 = Variable(input1.cuda())
            input2 = Variable(input2.cuda())

            if args.reid == "BtoB":
                if args.reid == "unimodal" and args.reid=="TtoT":
                    input1 = input2
                if args.reid == "unimodal" and args.reid=="VtOV":
                    input1 = input1
                test_mode=0
                feat_pool, feat_fc = net(input1, input2, modal=test_mode, fuse=args.fuse , modality=modality)
            # If we want to test cross modal reid with our multi modal models, keep those elifs
            elif args.reid == "VtoT" or args.reid == "TtoT":
                test_mode = 2
                feat_pool, feat_fc = net(input2, input2, modal=test_mode, fuse = args.fuse)
            elif args.reid == "TtoV" or args.reid == "VtoV":
                test_mode = 1
                feat_pool, feat_fc = net(input1, input1, modal=test_mode, fuse = args.fuse)

            # print(feat_pool.shape)
            # print(feat_fc.shape)
            query_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            query_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
        print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    return query_feat_pool, query_feat_fc

#
# if args.fusion == "late":


end = time.time()

if args.dataset == "regdb":
    nclass = 164
    data_path = '../Datasets/RegDB/'
    net = []
    net2 = [[],[],[],[],[]]
    Networks = {"early": Network_early(nclass).to(device), "layer1": Network_layer1(nclass).to(device), \
                "layer2": Network_layer2(nclass).to(device),
                "layer3": Network_layer3(nclass).to(device), \
                "layer4": Network_layer4(nclass).to(device),
                "layer5": Network_layer5(nclass).to(device), \
                "unimodal": Network_unimodal(nclass).to(device),
                "late": Network_unimodal(nclass).to(device)}
    for k in range(5):
        suffix = f'RegDB_{args.reid}_fuseType({args.fuse})_person_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_fold_{k}'
        print('==> Resuming from checkpoint..')
        model_path = checkpoint_path + suffix + '_best.t'

        if args.fusion == "late" :
            suffix = f'RegDB_VtoV_fuseType(none)_person_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_fold_{k}'
            suffix2 = f'RegDB_TtoT_fuseType(none)_person_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_fold_{k}'
            model_path = checkpoint_path + suffix + '_best.t'

            model_path2 = checkpoint_path + suffix2 + '_best.t'
            print(f"model path2 : {model_path2}")

        print(f"model path : {model_path}")
        if os.path.isfile(model_path):
            print('==> loading checkpoint')
            checkpoint = torch.load(model_path)
            net.append(Networks[args.fusion])
            net[k].load_state_dict(checkpoint['net'])
            print(f"Fold {k} loaded")
            if args.fusion == "late" :
                if os.path.isfile(model_path2) :
                    print('==> loading checkpoint 2')
                    checkpoint2 = torch.load(model_path2)
                    net2[k] = Networks[args.fusion]
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

        query_feat_pool, query_feat_fc = extract_query_feat(query_loader, nquery = nquery, net = net[test_fold])
        gall_feat_pool,  gall_feat_fc = extract_gall_feat(gall_loader, ngall = ngall, net = net[test_fold])

        if args.fusion == "late":
            # Extraction for the IR images with the model trained on IR modality
            query_feat_pool2, query_feat_fc2 = extract_query_feat(query_loader, nquery=nquery, net=net2[test_fold], modality = "thermal")
            gall_feat_pool2, gall_feat_fc2 = extract_gall_feat(gall_loader, ngall=ngall, net=net2[test_fold], modality = "thermal")

            # Basic summation of FC normalized output (should be the prob for each class )
            query_feat_fc = query_feat_fc + query_feat_fc2
            gall_feat_fc = gall_feat_fc + gall_feat_fc2

            # # Normalisation
            # norm = query_feat_fc.pow(2).sum(1, keepdim=True).pow(1. / 2)
            # query_feat_fc = query_feat_fc.div(norm)
            # norm = gall_feat_fc.pow(2).sum(1, keepdim=True).pow(1. / 2)
            # gall_feat_fc = gall_feat_fc.div(norm)

        # pool5 feature
        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        cmc_pool, mAP_pool, mINP_pool = eval_regdb(-distmat_pool,query_label , gall_label)

        # fc feature
        distmat = np.matmul( query_feat_fc, np.transpose(gall_feat_fc))
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


if args.dataset == 'sysu':
    nclass = 316
    data_path = '../Datasets/SYSU/'
    net = []
    # Since we have 5 folds max, this loop search for the 5 potentially saved models
    for k in range(5):
        suffix = f'SYSU_{args.reid}_fuseType({args.fuse})_person_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_fold_{k}'

        print('==> Resuming from checkpoint..')
        model_path = checkpoint_path + suffix + '_best.t'
        print(f"model path : {model_path}")
        if os.path.isfile(model_path):
            print('==> loading checkpoint')

            checkpoint = torch.load(model_path)
            Networks = {"early": Network_early(nclass).to(device), "layer1": Network_layer1(nclass).to(device), \
                        "layer2": Network_layer2(nclass).to(device), "layer3": Network_layer3(nclass).to(device), \
                        "layer4": Network_layer4(nclass).to(device), "layer5": Network_layer5(nclass).to(device), \
                        "unimodal": Network_unimodal(nclass).to(device)}
            net.append(Networks[args.fusion])

            # Append the found model in the network list
            net[k].load_state_dict(checkpoint['net'])
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
        query_feat_pool, query_feat_fc = extract_query_feat(query_loader, nquery=nquery, net=net[test_fold])
        gall_feat_pool, gall_feat_fc = extract_gall_feat(gall_loader,ngall = ngall, net = net[test_fold])

        # pool5 feature
        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        cmc_pool, mAP_pool, mINP_pool = eval_sysu(-distmat_pool, query_label, gall_label, query_cam, gall_cam)

        # fc feature
        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
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

# Means
cmc = all_cmc / loaded_folds
mAP = all_mAP / loaded_folds
mINP = all_mINP / loaded_folds

cmc_pool = all_cmc_pool / loaded_folds
mAP_pool = all_mAP_pool / loaded_folds
mINP_pool = all_mINP_pool / loaded_folds
print('All Average:')
print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))

for k in range(len(cmc)):
    writer.add_scalar('cmc curve', cmc[k]*100, k + 1)