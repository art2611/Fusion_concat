import torch
import sys
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import time
from data_loader import *
from torchvision import transforms
from utils import IdentitySampler, AverageMeter, adjust_learning_rate
from loss import BatchHardTripLoss
from tensorboardX import SummaryWriter

from model import Global_network
from evaluation import *
import argparse
from datetime import date

parser = argparse.ArgumentParser(description='PyTorch Multi-Modality Training')
parser.add_argument('--fusion', default='fc_fuse', help='Which layer to fuse (early, layer1, layer2 .., layer5, fc_fuse, gmu, unimodal)')
parser.add_argument('--fuse', default='fc_fuse', help='Fusion type (cat / cat_channel / sum / fc_fuse / gmu)')
parser.add_argument('--fold', default='0', help='Fold number (0 to 4)')
parser.add_argument('--dataset', default='TWorld', help='dataset name (RegDB / SYSU )')
parser.add_argument('--reid', default='BtoB', help='Type of ReID (BtoB / VtoV / TtoT)')
parser.add_argument('--LOO', default='query', help='Leave one out (query / gallery)')

args = parser.parse_args()

# Function to extract gallery features
def extract_gall_feat(gall_loader, ngall, net):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    # For resnet50
    # gall_feat_pool = np.zeros((ngall, 2048))
    # gall_feat_fc = np.zeros((ngall, 2048))
    # For resnet18
    gall_feat_pool = np.zeros((ngall, 512))
    gall_feat_fc = np.zeros((ngall, 512))

    with torch.no_grad():
        for batch_idx, (input1, input2, label) in enumerate(gall_loader):
            batch_num = input1.size(0)
            input1 = Variable(input1.cuda())
            input2 = Variable(input2.cuda())

            #Test mode 0 by default if BtoB and we need to use both inputs
            feat_pool, feat_fc, _ = net(input1, input2, fuse=args.fuse, modality=args.reid)

            gall_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            gall_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    return gall_feat_pool, gall_feat_fc

#Function to extract query image features
def extract_query_feat(query_loader, nquery, net):
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, 512))
    query_feat_fc = np.zeros((nquery, 512))

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

            # Test mode 0 by default if BtoB and we need to use both inputs
            feat_pool, feat_fc, _ = net(input1, input2, fuse=args.fuse, modality=args.reid)
            # print(feat_pool.shape)
            # print(feat_fc.shape)
            query_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            query_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
        print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    return query_feat_pool, query_feat_fc





device = 'cuda' if torch.cuda.is_available() else 'cpu'

### Tensorboard init - This is used afterwards to plot results on local website
today = date.today()
# dd/mm/YY
d1 = today.strftime("%d")
writer = SummaryWriter(f"runs/{args.dataset}_{args.reid}_{args.fusion}_LOO_{args.LOO}_Fusion_train_fusiontype({args.fuse})_{args.dataset}_day{d1}_{time.time()}")

### Verify the fusion args is good
fusion_list=['early', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'fc_fuse', 'gmu', 'unimodal']
fuse_list=['cat', 'cat_channel', 'sum', 'fc_fuse', 'gmu', 'none']
if args.fusion not in fusion_list :
    sys.exit(f'--fusion should be in {fusion_list}')
if args.fuse not in fuse_list :
    sys.exit(f'--fuse should be in {fuse_list}')

# Init variables :
img_w = 144
img_h = 288
test_batch_size = 64
batch_num_identities = 8 # 8 different identities in a batch
num_of_same_id_in_batch = 4 # Number of same identity in a batch
workers = 4
lr = 0.1
checkpoint_path = '../save_model/'


# Data info  :

# class ImgAugTransform:
#     def __init__(self):
#         self.aug = iaa.Sequential([
#             iaa.Scale((224, 224)),
#             iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
#             iaa.Fliplr(0.5),
#             iaa.Affine(rotate=(-20, 20), mode='symmetric'),
#             iaa.Sometimes(0.25,
#                           iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
#                                      iaa.CoarseDropout(0.1, size_percent=0.5)])),
#             iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
#         ])
#
#     def __call__(self, img):
#         img = np.array(img)
#         return self.aug.augment_image(img)
# transforms = ImgAugTransform()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
if args.dataset=="RegDB" :
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(10),
        transforms.RandomCrop((img_h, img_w)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomPerspective(distortion_scale=0.25, p=0.5, interpolation=2),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
        #The following lines has to be removed for a data visualisation
        transforms.ToTensor(),
        normalize,
    ])

else :
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

######################################### TRAINING SET

data_path = f'../Datasets/{args.dataset}/'
if args.LOO == "query" :
    suffix = f'{args.dataset}_{args.reid}_fuseType({args.fuse})_{args.fusion}person_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_fold_{args.fold}'
else :
    suffix = f'{args.dataset}_{args.reid}_fuseType({args.fuse})_{args.fusion}person_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_fold_{args.fold}_LOO_{args.LOO}'

trainset = TrainingData(data_path, args.dataset, transform_train, args.fold)

# if args.dataset == 'sysu':
#     data_path = '../Datasets/SYSU/'
#     suffix = f'SYSU_{args.reid}_fuseType({args.fuse})_person_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_fold_{args.fold}'
#     # training set
#     trainset = SYSUData(data_path, transform=transform_train, fold = args.fold)
#
# elif args.dataset == 'RegDB':
#     data_path = '../Datasets/RegDB/'
#     suffix = f'RegDB_{args.reid}_fuseType({args.fuse})_person_fusion({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_fold_{args.fold}'
#     trainset = RegDBData(data_path, transform=transform_train, fold = args.fold)


#The following lines can be used in a way to visualize data and transformations
# w=0
# for i in range(0, 24):
#     w += 1
#     print(i)
#     plt.subplot(5,5,w)
#     plt.imshow(transform_train(trainset.train_thermal_image[i]))
# plt.show()
# sys.exit()

# Get ids positions (color_pos[0] returns [0,1,...,9] which are the positions of id 0 in trainset.train_color_image)
color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)


######################################### VALIDATION SET

# Validation imgs and labels, depending of the cross validation fold
query_img, query_label, query_cam,\
    gall_img, gall_label, gall_cam = process_data(data_path, "valid", args.dataset, LOO = args.LOO, fold =args.fold)

# Gallery and query set
gallset = Prepare_set(gall_img, gall_label, transform=transform_test, img_size=(img_w, img_h))
queryset = Prepare_set(query_img, query_label, transform=transform_test, img_size=(img_w, img_h))
# Validation data loader
gall_loader = torch.utils.data.DataLoader(gallset, batch_size= test_batch_size, shuffle=False, num_workers= workers)
query_loader = torch.utils.data.DataLoader(queryset, batch_size= test_batch_size, shuffle=False, num_workers= workers)

# Init variables
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

# Networks = {"early":Network_early(n_class).to(device),"layer1":Network_layer1(n_class).to(device), \
#             "layer2":Network_layer2(n_class).to(device), "layer3":Network_layer3(n_class).to(device), \
#             "layer4":Network_layer4(n_class).to(device), "layer5":Network_layer5(n_class).to(device), \
#             "unimodal":Network_unimodal(n_class).to(device), "late":Network_unimodal(n_class).to(device)}

# Just call the network needed - Two distinct model if the fusion is at scores position
# net = Networks[args.fusion]
Fusion_layer = {"early": 0,"layer1":1, "layer2":2, "layer3":3, "layer4":4, "layer5":5, "fc_fuse":5, "gmu" : 5, "unimodal":0}

# New global model
net = Global_network(n_class, fusion_layer=Fusion_layer[args.fusion]).to(device)

######################################### TRAIN AND VALIDATION FUNCTIONS

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
        # Labels 1 and 2 are the same because the two inputs correspond to the same identity
        labels = label1

        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())
        labels = Variable(labels.cuda())

        data_time.update(time.time() - end)

        # If the reid is unimodal VtoV (Visible to Visible), the network use only the first input
        # If the reid is unimodal TtoT (Thermal to Thermal), the network use only the second input
        # If the reid is using both images (BtoB), the network uses the two inputs

        feat, out0, = net(input1, input2, fuse = args.fuse, modality=args.reid)

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
    # For each batch, write in tensorBoard :
    writer.add_scalar('Loss training', train_loss.avg, epoch)
    writer.add_scalar('Accuracy training', 100. * correct / total, epoch)

def valid(epoch):
    # Get the timer
    end = time.time()

    #Get all normalized distance
    gall_feat_pool, gall_feat_fc = extract_gall_feat(gall_loader, n_gall, net = net)
    query_feat_pool, query_feat_fc = extract_query_feat(query_loader, n_query, net = net)

    print(f"Feature extraction time : {time.time() - end}")
    start = time.time()

    # compute the similarity (cosine)
    distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
    distmat_fc = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

    # evaluation
    # if args.dataset == 'RegDB'or args.dataset == 'TWorld' or args.dataset == "SYSU" :
    if args.dataset == 'RegDB'or args.dataset == 'TWorld' :
        cmc, mAP, mINP = evaluation(-distmat_fc, query_label, gall_label, dataset = args.dataset, LOO = args.LOO)
        cmc_att, mAP_att, mINP_att  = evaluation(-distmat_pool, query_label, gall_label, dataset = args.dataset, LOO = args.LOO)

    elif args.dataset == 'SYSU':
        cmc, mAP, mINP = eval_sysu(-distmat_fc, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_pool, query_label, gall_label, query_cam, gall_cam)

    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))
    writer.add_scalar('mAP validation', mAP, epoch)

    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att

######################################### TRAINING

print('==> Start Training...')
#Train function
ignored_params = list(map(id, net.bottleneck.parameters())) \
                 + list(map(id, net.fc.parameters())) + list(map(id, net.fc_fuse.parameters()))
# ignored_params = list(map(id, net.bottleneck.parameters())) \
#                  + list(map(id, net.fc.parameters()))


base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

optimizer = optim.SGD([
    {'params': base_params, 'lr': 0.1 * lr},
    {'params': net.bottleneck.parameters(), 'lr': lr},
    {'params': net.fc_fuse.parameters(), 'lr': lr},
    {'params': net.fc.parameters(), 'lr': lr}],
    weight_decay=5e-4, momentum=0.9, nesterov=True)

loader_batch = batch_num_identities * num_of_same_id_in_batch
# Loss functions
criterion_id = nn.CrossEntropyLoss().to(device)
criterion_tri = BatchHardTripLoss(batch_size=loader_batch, margin= 0.3).to(device)

best_map = 0
training_time = time.time()

if args.dataset == "RegDB" :
    epoch_number = 81
else :
    epoch_number = 61

for epoch in range(epoch_number):

    print('==> Preparing Data Loader...')
    # identity sampler - Give iteratively index from a randomized list of color index and thermal index
    sampler = IdentitySampler(trainset.train_color_label, trainset.train_thermal_label, \
                              color_pos, thermal_pos, num_of_same_id_in_batch, batch_num_identities, args.dataset)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    # print(epoch)
    # print(trainset.cIndex)
    # print(trainset.tIndex)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=loader_batch, \
                            sampler=sampler, num_workers=workers, drop_last=True)
    # print(len(trainloader))

    # training
    train(epoch)

    # Call the validation part every two epochs
    if epoch > 0 and epoch % 2 == 0  :
        print(f'Test Epoch: {epoch}')

        # testing
        cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = valid(epoch)
        # save model
        # if cmc[0] > best_acc:
        # USE OF THE MAP INSTEAD OF THE CMC[0] BECAUSE  CMC HIT 100% with REGDB DATASET
        if mAP > best_map:
            # best_acc = cmc_att[0]
            best_map = mAP
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'cmc': cmc_att,
                'mAP': mAP_att,
                'mINP': mINP_att,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')

        # # save model
        # if epoch > 10 and epoch % 20 == 0:
        #     state = {
        #         'net': net.state_dict(),
        #         'cmc': cmc,
        #         'mAP': mAP,
        #         'epoch': epoch,
        #     }
        #     torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))

        # print(
        #     'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        #         cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        # print(
        #     'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        #         cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        print('fc : Rank-1: {:.2%} | Rank-5: {:.2%} | mAP: {:.2%}| mINP: {:.2%}'.format(cmc[0], cmc[4], mAP, mINP))
        print('att : Rank-1: {:.2%} | Rank-5: {:.2%} | mAP: {:.2%}| mINP: {:.2%}'.format(cmc_att[0], cmc_att[4], mAP_att, mINP_att))
        print('Best Epoch [{}]'.format(best_epoch))
        if args.fusion =="unimodal" :
            print(f' Training {args.fusion} {args.reid} - {args.fuse} fusion - fold number ({args.fold})')
        else :
            print(f' Training {args.fusion} - {args.fuse} fusion - fold number ({args.fold})')


print(f' Training time for {args.fusion} {args.fuse} fusion : {time.time() - training_time}')

