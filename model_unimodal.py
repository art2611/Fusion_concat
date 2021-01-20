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

class Network_unimodal(nn.Module):
    def __init__(self,  class_num, arch='resnet50'):
        super(Network_unimodal, self).__init__()

        model_unimodal = resnet50(pretrained=True)
        # avg pooling to global pooling
        self.unimodal = model_unimodal

        pool_dim = 2048

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.fc = nn.Linear(pool_dim, class_num, bias=False)
        self.l2norm = Normalize(2)

    def forward(self, x, x1, modal="no use here", fuse="no use here"):
        x = self.unimodal.conv1(x)
        x = self.unimodal.bn1(x)
        x = self.unimodal.relu(x)
        x = self.unimodal.maxpool(x)
        x = self.unimodal.layer1(x)
        x = self.unimodal.layer2(x)
        x = self.unimodal.layer3(x)
        x = self.unimodal.layer4(x)

        x_pool = self.avgpool(x)
        x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        feat = self.bottleneck(x_pool) #torch.Size([64, 2048])

        if self.training:
            return x_pool, self.fc(feat)
        else:
            return self.l2norm(x_pool), self.l2norm(feat)


#   from torchsummary import summary
#   model = Network_layer5(250, arch='resnet50')
#   summary(model, [(3, 288, 144),(3, 288, 144)] , batch_size=32)


#   model = Network(250, arch='resnet50')
#   print(resneut50(pretrained= True))
#   print(thermal_module())