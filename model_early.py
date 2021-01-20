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

class shared_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(shared_resnet, self).__init__()

        model_base = resnet50(pretrained=True)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_base.fc = Identity()
        self.base = model_base

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

class Network_early(nn.Module):
    def __init__(self,  class_num, arch='resnet50'):
        super(Network_early, self).__init__()

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
            if fuse == "cat":
                x = torch.cat((x1, x2), -1)
            elif fuse == "sum":
                x = x1.add(x2)
        elif modal == 1:
            x = x1 #Visible
        elif modal == 2:
            x = x2 #Thermal

        x = self.shared_resnet(x)

        x_pool = self.avgpool(x)
        x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        feat = self.bottleneck(x_pool) #torch.Size([64, 2048])

        if self.training:
            return x_pool, self.fc(feat)
        else:
            return self.l2norm(x_pool), self.l2norm(feat)


# from torchsummary import summary
# model = Network_layer1(250, arch='resnet50')
# summary(model, [(3, 288, 144),(3, 288, 144)] , batch_size=32)

#print(resneut50(pretrained= True))
# print(thermal_module())