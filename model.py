import torch
import torch.nn as nn
from torch.nn import init
# from torchvision.models import resnet50
from torchvision.models import resnet18 as resnet50
# from torchsummary import summary
import torch.nn.functional as F
# import theano
# from blocks import initialization
# from blocks.bricks import (Initializable, FeedforwardSequence, LinearMaxout,
#                            Tanh, lazy, application, BatchNormalization, Linear,
#                            NDimensionalSoftmax, Logistic, Softmax, Sequence, Rectifier)
# from blocks.bricks.parallel import Fork
# from blocks.utils import shared_floatx_nans
# from blocks.roles import add_role, WEIGHT

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Normalize_MinMax(nn.Module):
    def __init__(self):
        super(Normalize_MinMax, self).__init__()

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class visible_module(nn.Module):
    def __init__(self, fusion_layer=4, arch='resnet50'):
        super(visible_module, self).__init__()

        self.visible = resnet50(pretrained=True)

        self.fusion_layer = fusion_layer
        layer0 = [self.visible.conv1, self.visible.bn1, self.visible.relu, self.visible.maxpool]
        self.layer_dict = {"layer0": nn.Sequential(*layer0), "layer1":self.visible.layer1, "layer2":self.visible.layer2, "layer3":self.visible.layer3, "layer4":self.visible.layer4}

    def forward(self, x):
        for i in range(0, self.fusion_layer):
            x = self.layer_dict["layer" + str(i)](x)
        return x

class thermal_module(nn.Module):
    def __init__(self, fusion_layer=4, arch='resnet50'):
        super(thermal_module, self).__init__()

        self.thermal = resnet50(pretrained=True)

        self.fusion_layer = fusion_layer
        layer0 = [self.thermal.conv1, self.thermal.bn1, self.thermal.relu, self.thermal.maxpool]
        self.layer_dict = {"layer0": nn.Sequential(*layer0), "layer1":self.thermal.layer1, "layer2":self.thermal.layer2, "layer3":self.thermal.layer3, "layer4":self.thermal.layer4}
    def forward(self, x):
        for i in range(0, self.fusion_layer):
            x = self.layer_dict["layer" + str(i)](x)
        return x

class shared_resnet(nn.Module):
    def __init__(self, fusion_layer = 4, arch='resnet50'):
        super(shared_resnet, self).__init__()

        self.fusion_layer = fusion_layer

        model_base = resnet50(pretrained=True)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_base.fc = Identity()
        self.base = model_base

        layer0 = [self.base.conv1, self.base.bn1, self.base.relu, self.base.maxpool]
        self.layer_dict = {"layer0": nn.Sequential(*layer0), "layer1":self.base.layer1, "layer2":self.base.layer2, "layer3":self.base.layer3, "layer4":self.base.layer4}

    def forward(self, x):

        for i in range(self.fusion_layer, 5):
            x = self.layer_dict["layer" + str(i)](x)
        return x

class fusion_function_concat(nn.Module): # concat the features and
    def __init__(self, layer_size):
        super(fusion_function_concat, self).__init__()
        layers = [
            nn.Conv2d(2*layer_size, layer_size, kernel_size=7, stride=2, padding=3,
                      bias=False),
            nn.BatchNorm2d(layer_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]
        self.fusionBlock = nn.Sequential(*layers)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.fusionBlock(x)
        return x

class GatedBimodal(nn.Module):
    u"""Gated Multimodal Unit neural network - Bimodal use"""
    def __init__(self, dim):
        super(GatedBimodal, self).__init__()

        self.dim = dim

        #Activation functions
        self.activation = nn.Tanh()
        self.gate_activation = nn.Sigmoid()

        # Learnable weights definition - As describe in the paper
        self.hidden1 = nn.Linear(dim, dim, bias=False)
        self.hidden2 = nn.Linear(dim, dim, bias=False)
        self.hidden_sigmoid = nn.Linear(dim*2, 1, bias=False)


    # x1 and x2 will respectively be RGB input and IR thermal input.
    def forward(self, x1, x2):


        h1 = self.activation(self.hidden1(x1)) # torch.Size([batch size , feature size])
        h2 = self.activation(self.hidden1(x2)) # torch.Size([batch size , feature size])
        x = torch.cat((h1, h2), dim=1)
        z = self.gate_activation(self.hidden_sigmoid(x)) # torch.Size([batch size , 1])

        return z.view(z.size()[0],1)*h1 + (1-z).view(z.size()[0],1)*h2, z

        # # Prepare the cat tensor for incoming z calcul
        # x = torch.cat((x1, x2), 1) # torch.Size([batch size, 2 * dim of input features])
        # # Get the batch size
        # batch_size = x.shape[0]
        # # print(self.Wz)
        # # Get vector of scalar. One scalar for each feature from the batch
        # hv = self.activation(torch.mm(self.Wv, torch.transpose(x1, 0, 1))) # torch.Size([1, batch size])
        # ht = self.activation(torch.mm(self.Wt, torch.transpose(x2, 0, 1))) # torch.Size([1, batch size])
        # # print(hv)
        # # Get the weights for weighted sum fusion of the two modalities
        # z = self.gate_activation(torch.mm(x, self.Wz)) # torch.Size([batch size, dim of input features])
        #
        # # Prepare the fused feature tensor of size [batch size , dim input feature)
        # fused_feat = torch.rand(batch_size, self.dim).cuda()
        #
        # # For each feature from batch, return the weighted sum
        # for k in range(batch_size) :
        #     fused_feat[k] = z[k][:]*hv[0][k] + (1-z[k][:])*ht[0][k]  # torch.Size([1, dim of input feature])
        #
        # # Get the fused features and the matrix of weight, in a way to see which modality contributed the most further
        # return fused_feat, z

class Global_network(nn.Module):
    def __init__(self,  class_num, arch='resnet50', fusion_layer=4):
        super(Global_network, self).__init__()

        self.thermal_module = thermal_module(arch=arch, fusion_layer=fusion_layer)
        self.visible_module = visible_module(arch=arch, fusion_layer=fusion_layer)

        # self.convolution_after_fuse = torch.nn.Conv2d(2048, 1024, 1)
        resnet18_layer_size = [3, 64, 64, 128, 256, 512]

        self.fusion_function_concat = fusion_function_concat(resnet18_layer_size[fusion_layer])
        # pool_dim = 2048
        #Resnet18 pool dim
        pool_dim = 512

        # self.gbu = GatedBimodal(pool_dim, weights_init=initialization.Uniform(width=gbu_init_range))

        self.shared_resnet = shared_resnet(arch=arch, fusion_layer=fusion_layer)


        # self.bottleneck.apply(weights_init_kaiming)
        # self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.gmu = GatedBimodal(pool_dim)
        self.fc_fuse = nn.Sequential(nn.Linear(2*pool_dim, pool_dim, bias = True), nn.ReLU())
        # self.fc_fuse = nn.Linear(2*pool_dim, pool_dim, bias = True)

        self.fc = nn.Linear(pool_dim, class_num, bias=False)
        self.l2norm = Normalize(2)

    def forward(self, x1, x2, modal=0, fuse="cat_channel", modality = "BtoB"):
        if fuse != "none":
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            # Multiple fusion definitions
            if fuse == "cat" :
                x = torch.cat((x1, x2), -1)
            elif fuse == "sum":
                x = x1.add(x2)
            elif fuse == "cat_channel" :
                x = self.fusion_function_concat(x1, x2)
                #The fc can't be used here since the dim is not already [32,1024] but [32,1024,7,2]
            elif fuse == "fc_fuse" or fuse == "gmu":
                x = x1
        # If fuse == none : we train a unimodal model => RGB or IR ? Refer to modality
        else :
            if modality=="VtoV" :
                x = self.visible_module(x1)
            elif modality=="TtoT" :
                x = self.visible_module(x2)

        x = self.shared_resnet(x)


        x_pool = self.avgpool(x)
        x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))
        # The fc can be used here since the dim is ok but it is less working than after the batch norm


        if fuse == "fc_fuse" or fuse == "gmu":
            x_pool2 = self.avgpool2(x2)
            x_pool2 = x_pool.view(x_pool2.size(0), x_pool2.size(1))

            # Normalization
            x_pool = self.l2norm(x_pool)
            x_pool2 = self.l2norm(x_pool2)
            if fuse == "gmu":

                x_pool, z = self.gmu(x_pool, x_pool2)
            elif fuse == "fc_fuse" :
                x_pool = torch.cat((x_pool, x_pool2), 1)
                x_pool = self.fc_fuse(x_pool)
                x_pool = F.relu(x_pool)

        feat = self.bottleneck(x_pool)

        if self.training:
            return x_pool, self.fc(feat)
        else:
            return self.l2norm(x_pool), self.l2norm(feat), feat

class Gated_classifier(nn.Module):
    def __init__(self, class_num):
        super(Gated_classifier, self).__init__()

        # self.gmu = GatedBimodal(pool_dim)
        pool_dim = 512
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.fc = nn.Linear(pool_dim, class_num, bias=False)
        self.l2norm = Normalize(2)
    def forward(self, x1, x2):

        x_pool, z = self.gmu(x1, x2)

        feat = self.bottleneck(x_pool)

        if self.training:
            return x_pool, self.fc(feat)
        else:
            return self.l2norm(x_pool), self.l2norm(feat), feat


# from torchsummary import summary
# model = Global_network(250, arch='resnet50', fusion_layer=5)
# summary(model, [(3, 288, 144),(3, 288, 144)] , batch_size=32)
#
# print(Network_layer4(250))
# print(thermal_module())