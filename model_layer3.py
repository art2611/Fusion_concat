import torch
import torch.nn as nn
from torch.nn import init
from torchvision.models import resnet50


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

class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_t = resnet50(pretrained=True)
        # avg pooling to global pooling
        self.visible = model_t


    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        x = self.visible.layer1(x)
        x = self.visible.layer2(x)
        return x

class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x = self.thermal.layer1(x)
        x = self.thermal.layer2(x)

        return x

class shared_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(shared_resnet, self).__init__()

        model_base = resnet50(pretrained=True)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_base.fc = Identity()
        self.base = model_base

    def forward(self, x):
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

class Network_layer3(nn.Module):
    def __init__(self,  class_num, arch='resnet50'):
        super(Network_layer3, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.shared_resnet = shared_resnet(arch=arch)

        pool_dim = 2048

        # self.bottleneck.apply(weights_init_kaiming)
        # self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.fc = nn.Linear(pool_dim, class_num, bias=False)
        self.l2norm = Normalize(2)

    def forward(self, x1, x2, modal=0, fuse="sum"):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            if fuse == "cat":
                x = torch.cat((x1, x2), -1)
            elif fuse == "sum":
                x = x1.add(x2)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        x = self.shared_resnet(x)

        x_pool = self.avgpool(x)
        x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        feat = self.bottleneck(x_pool) #torch.Size([64, 2048])

        if self.training:
            return x_pool, self.fc(feat)
        else:
            return self.l2norm(x_pool), self.l2norm(feat)

# from torchsummary import summary
# model = Network_layer3(250, arch='resnet50')
# summary(model, [(3, 288, 144),(3, 288, 144)] , batch_size=32)
# model = Network(250, arch='resnet50')
#print(resneut50(pretrained= True))
# print(thermal_module())