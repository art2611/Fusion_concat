import torch
import sys
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import time
from data_loader import *
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from utils import IdentitySampler, AverageMeter, adjust_learning_rate
from loss import BatchHardTripLoss
from tensorboardX import SummaryWriter
from model_layer1 import Network_layer1
from model_layer2 import Network_layer2
from model_layer3 import Network_layer3
from model_layer4 import Network_layer4
from model_layer5 import Network_layer5
from model_unimodal import Network_unimodal
from model_early import Network_early
from multiprocessing import freeze_support
# from test import extract_gall_feat, extract_query_feat
from evaluation import *
import argparse
from datetime import date

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--fusion', default='layer1', help='layer to fuse')
parser.add_argument('--fold', default='0', help='Fold number, from 0 to 4')
parser.add_argument('--dataset', default='regdb', help='dataset name: regdb or sysu')
parser.add_argument('--reid', default='VtoT', help='Visible to thermal reid')
parser.add_argument('--split', default='paper_based', help='How to split data')
args = parser.parse_args()


def extract_gall_feat(gall_loader, ngall, net):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0

    gall_feat_pool = np.zeros((ngall, 2048))
    gall_feat_fc = np.zeros((ngall, 2048))

    with torch.no_grad():
        for batch_idx, (input1, input2, label) in enumerate(gall_loader):
            batch_num = input1.size(0)
            input1 = Variable(input1.cuda())
            input2 = Variable(input2.cuda())
            if args.fusion=="unimodal" or args.reid == "BtoB":
                if args.reid== "TtoT" :
                    input1 = input2
                #Test mode 0 by default if BtoB
                feat_pool, feat_fc = net(input1, input1)
            elif args.reid == "VtoT" or args.reid == "TtoT":
                test_mode = 2
                feat_pool, feat_fc = net(input2, input2, modal=test_mode)
            elif args.reid == "TtoV" or args.reid == "VtoV":
                test_mode = 1
                feat_pool, feat_fc = net(input1, input1, modal=test_mode)

            gall_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            gall_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    return gall_feat_pool, gall_feat_fc

def extract_query_feat(query_loader, nquery, net):
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0

    query_feat_pool = np.zeros((nquery, 2048))
    query_feat_fc = np.zeros((nquery, 2048))

    with torch.no_grad():
        for batch_idx, (input1, input2, label) in enumerate(query_loader):
            batch_num = input1.size(0)
            # print(f"batch num : {batch_num}")
            # print(input1.size(0))
            # print(input2.size(0))
            # print(label)
            # print(batch_idx)
            input1 = Variable(input1.cuda())
            input2 = Variable(input2.cuda())
            if args.fusion=="unimodal" or args.reid == "BtoB":
                if args.reid == "TtoT":
                    input1 = input2
                feat_pool, feat_fc = net(input1, input1)
            elif args.reid == "VtoT" or args.reid == "TtoT":
                test_mode = 2
                feat_pool, feat_fc = net(input2, input2, modal=test_mode)
            elif args.reid == "TtoV" or args.reid == "VtoV":
                test_mode = 1
                feat_pool, feat_fc = net(input1, input1, modal=test_mode)
            # print(feat_pool.shape)
            # print(feat_fc.shape)
            query_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            query_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
        print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    return query_feat_pool, query_feat_fc

def multi_process() :
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    split_list = ["paper_based", "experience_based"]
    if args.split not in split_list:
        sys.exit(f"--split should be in {split_list}")

    ### Tensorboard init
    today = date.today()
    # dd/mm/YY
    d1 = today.strftime("%d")
    writer = SummaryWriter(f"runs/{args.reid}_{args.fusion}_Fusion_train_{args.dataset}_day{d1}_{time.time()}")

    ### assure good fusion args
    fusion_list=['early', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5', "unimodal"]
    if args.fusion not in fusion_list :
        sys.exit(f'--fusion should be in {fusion_list}')

    # Init variables :
    img_w = 144
    img_h = 288
    test_batch_size = 64
    batch_num_identities = 8 # 8 different identities in a batch
    num_of_same_id_in_batch = 4 # Number of same identity in a batch
    workers = 4
    lr = 0.001
    checkpoint_path = '../save_model/'


    # Data info  :

    #  log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(10),
        transforms.RandomCrop((img_h, img_w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        normalize,
    ])

    Timer1 = time.time()

    if args.dataset == 'sysu':
        data_path = '../Datasets/SYSU/'
        suffix = f'SYSU_{args.reid}_person_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_fold_{args.fold}'
    elif args.dataset == 'regdb':
        data_path = '../Datasets/RegDB/'
        suffix = f'RegDB_{args.reid}_person_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}'

    ######################################### TRAIN SET

    if args.dataset == 'sysu':
        # training set
        trainset = SYSUData_clean(data_path, transform=transform_train, fold = args.fold)
        # trainset = SYSUData(data_path, transform=transform_train)

        # generate the idx of each person identity
        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
        # Validation set
        if args.reid == "BtoB" or args.fusion == "unimodal":
            query_img, query_label, query_cam, gall_img, gall_label, gall_cam = \
                process_BOTH_sysu(data_path, "valid", fold = args.fold)
        else :
            query_img, query_label, query_cam = process_query_sysu(data_path, "valid", mode="all", trial=0, reid=args.reid)
            gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, "valid", mode="all", trial=0, reid=args.reid)

    elif args.dataset == 'regdb':
        trainset = RegDBData_clean(data_path, trial = 1, transform=transform_train, fold = 0)
        # trainset = RegDBData(data_path, trial = 1, transform=transform_train)
        # print(trainset.train_thermal_label)
        # print(trainset.train_color_label)
        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
        print(len(color_pos))
        query_img, query_label, gall_img, gall_label = process_test_regdb(data_path, trial=1, modal=args.reid, split=args.split)


    ######################################### VALID SET
    # Gallery and query set
    if args.reid == "BtoB" or args.fusion == "unimodal":
        gallset = TestData_both(gall_img, gall_label, transform=transform_test, img_size=(img_w, img_h))
        queryset = TestData_both(query_img, query_label, transform=transform_test, img_size=( img_w, img_h))
    else :
        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(img_w, img_h))
        queryset = TestData(query_img, query_label, transform=transform_test, img_size=( img_w, img_h))

    # Test data loader
    gall_loader = torch.utils.data.DataLoader(gallset, batch_size= test_batch_size, shuffle=False, num_workers= workers)
    query_loader = torch.utils.data.DataLoader(queryset, batch_size= test_batch_size, shuffle=False, num_workers= workers)

    n_class = len(np.unique(trainset.train_color_label))
    n_query = len(query_label)
    n_gall = len(gall_label)

    print(f'Dataset {args.dataset} statistics:')
    print('   set     |  Nb ids |  Nb img    ')
    print('  ------------------------------')
    print(f'  visible  | {n_class:5d} | {len(trainset.train_color_label):8d}')
    print(f'  thermal  | {n_class:5d} | {len(trainset.train_thermal_label):8d}')
    print('  ------------------------------')
    print(f'  query    | {len(np.unique(query_label)):5d} | {n_query:8d}')
    print(f'  gallery  | {len(np.unique(gall_label)):5d} | {n_gall:8d}')
    print('  ------------------------------')
    print(f'Data Loading Time:\t {time.time() - Timer1:.3f}')
    print(' ')
    print('==> Building model..')

    ######################################### MODEL

    if args.fusion == "early" :
        net = Network_early(n_class).to(device)
    elif args.fusion=="layer1" :
        net = Network_layer1(n_class).to(device)
    elif args.fusion == "layer2":
        net = Network_layer2(n_class).to(device)
    elif args.fusion == "layer3" :
        net = Network_layer3(n_class).to(device)
    elif args.fusion=="layer4" :
        net = Network_layer4(n_class).to(device)
    elif args.fusion == "layer5" :
        net = Network_layer5(n_class).to(device)
    elif args.fusion == "unimodal" :
        net = Network_unimodal(n_class).to(device)


    ######################################### TRAINING
    print('==> Start Training...')
    #Train function
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.fc.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * lr},
        {'params': net.bottleneck.parameters(), 'lr': lr},
        {'params': net.fc.parameters(), 'lr': lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)

    def train(epoch):

        current_lr = adjust_learning_rate(optimizer, epoch, lr=lr)
        train_loss = AverageMeter()
        id_loss = AverageMeter()
        tri_loss = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        correct = 0
        total = 0

        # switch to train mode
        net.train()
        end = time.time()

        for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):
            # labels = torch.cat((label1, label2), 0)
            labels = label1

            input1 = Variable(input1.cuda())
            input2 = Variable(input2.cuda())
            labels = Variable(labels.cuda())

            data_time.update(time.time() - end)
            if args.reid == "TtoT":
                input1 = input2
            feat, out0, = net(input1, input2)
            # print(feat)
            # print(out0)
            # # print(feat)
            # print("FLAAAAAG")
            # print(feat.shape)
            # print(out0.shape)
            # print(labels.shape)

            loss_ce = criterion_id(out0, labels)

            loss_tri, batch_acc = criterion_tri(feat, labels)
            correct += (batch_acc / 2)
            _, predicted = out0.max(1)
            correct += (predicted.eq(labels).sum().item() / 2)

            loss = loss_ce + loss_tri
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update P
            train_loss.update(loss.item(), 2 * input1.size(0))
            id_loss.update(loss_ce.item(), 2 * input1.size(0))
            tri_loss.update(loss_tri.item(), 2 * input1.size(0))
            total += labels.size(0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % 50 == 0:
                print(f'Epoch: [{epoch}][{batch_idx}/{len(trainloader)}] '
                      f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      f'lr:{current_lr:.3f} '
                      f'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                      f'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                      f'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                      f'Accu: {100. * correct / total:.2f}')
        # For each batch, write in tensorBoard

        # writer.add_scalar('id_loss', id_loss.avg, epoch)
        # writer.add_scalar('tri_loss', tri_loss.avg, epoch)
        # writer.add_scalar('lr', current_lr, epoch)
        writer.add_scalar('total_loss', train_loss.avg, epoch)
        writer.add_scalar('Accuracy training', 100. * correct / total, epoch)

    def valid(epoch):

        end = time.time()
        #Get all normalized distance
        gall_feat_pool, gall_feat_fc = extract_gall_feat(gall_loader, n_gall, net = net)
        query_feat_pool, query_feat_fc = extract_query_feat(query_loader, n_query, net = net)
        print(f"Feature extraction time : {time.time() - end}")
        start = time.time()
        # compute the similarity
        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        distmat_fc = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

        # evaluation
        if args.dataset == 'regdb':
            cmc, mAP, mINP = eval_regdb(-distmat_pool, query_label, gall_label)
            cmc_att, mAP_att, mINP_att  = eval_regdb(-distmat_fc, query_label, gall_label)

        elif args.dataset == 'sysu':
            cmc, mAP, mINP = eval_sysu(-distmat_pool, query_label, gall_label, query_cam, gall_cam)
            cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_fc, query_label, gall_label, query_cam, gall_cam)

        print('Evaluation Time:\t {:.3f}'.format(time.time() - start))
        writer.add_scalar('Accuracy validation', mAP, epoch)

        return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att

    # Training part
    # start_epoch = 0
    loader_batch = batch_num_identities * num_of_same_id_in_batch
    # define loss function
    criterion_id = nn.CrossEntropyLoss().to(device)
    criterion_tri = BatchHardTripLoss(batch_size=loader_batch, margin= 0.3).to(device)
    best_acc = 0
    # for epoch in range(start_epoch, 81 - start_epoch):
    training_time = time.time()
    for epoch in range(41):

        print('==> Preparing Data Loader...')
        # identity sampler - Give iteratively index from a randomized list of color index and thermal index
        sampler = IdentitySampler(trainset.train_color_label, \
                                  trainset.train_thermal_label, color_pos, thermal_pos, num_of_same_id_in_batch, batch_num_identities,
                                  epoch)

        trainset.cIndex = sampler.index1  # color index
        trainset.tIndex = sampler.index2  # thermal index
        # print(epoch)
        # print(trainset.cIndex)
        # print(trainset.tIndex)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=loader_batch, \
                                sampler=sampler, num_workers=workers, drop_last=True)
        print(len(trainloader))
        # training
        train(epoch)

        if epoch > 0 and epoch % 2 == 0  :
            print(f'Test Epoch: {epoch}')

            # testing
            cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = valid(epoch)
            # save model
            if cmc_att[0] > best_acc:  # not the real best for sysu-mm01
                best_acc = cmc_att[0]
                best_epoch = epoch
                state = {
                    'net': net.state_dict(),
                    'cmc': cmc_att,
                    'mAP': mAP_att,
                    'mINP': mINP_att,
                    'epoch': epoch,
                }
                torch.save(state, checkpoint_path + suffix + '_best.t')

            # save model
            if epoch > 10 and epoch % 20 == 0:
                state = {
                    'net': net.state_dict(),
                    'cmc': cmc,
                    'mAP': mAP,
                    'epoch': epoch,
                }
                torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))

            print(
                'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
            print(
                'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
            print('Best Epoch [{}]'.format(best_epoch))

    print(f' Training time for {args.fusion} fusion : {time.time() - training_time}')

if __name__ == '__main__':
    freeze_support()
    multi_process()