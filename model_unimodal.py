import torch
import torch.nn as nn
from torch.nn import init
# from torchvision.models import resnet50
from torchvision.models import resnet18 as resnet50


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

        # Get the pretrained resnet50 model
        self.unimodal = resnet50(pretrained=True)

        # pool_dim = 2048
        pool_dim = 512

        # Prepare last layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.fc = nn.Linear(pool_dim, class_num, bias=False)
        self.l2norm = Normalize(2)

    def forward(self, x, x1, modal="no use here", fuse="no use here", modality = "visible"):
        # Firsts Resnet50 5 convolutional layers
        # x input shape : torch.Size([batch_size, 3, 288, 144])
        if modality == "thermal" :
            x = x1
        x = self.unimodal.conv1(x)      # Output : x.shape = torch.Size([batch_size, 64, 144, 72])
        x = self.unimodal.bn1(x)
        x = self.unimodal.relu(x)
        x = self.unimodal.maxpool(x)    # Output : x.shape = torch.Size([batch_size, 64, 72, 36])
        x = self.unimodal.layer1(x)     # Output : x.shape = torch.Size([batch_size, 256, 72, 36])
        x = self.unimodal.layer2(x)     # Output : x.shape = torch.Size([batch_size, 512, 36, 18])
        x = self.unimodal.layer3(x)     # Output : x.shape = torch.Size([batch_size, 1024, 18, 9])
        x = self.unimodal.layer4(x)     # Output : x.shape = torch.Size([batch_size, 2048, 9, 5])

        # AVG pooling
        x_pool = self.avgpool(x)
        x_pool = x_pool.view(x_pool.size(0), x_pool.size(1)) # Output : x.shape = torch.Size([batch_size, 2048])

        #BatchNorm1d
        feat = self.bottleneck(x_pool)

        # If training, return features (x_pool) for the triplet loss and the scores (fc(feat)) for the cross entropy loss
        if self.training:
            return x_pool, self.fc(feat)
        # Else (testing) : return normalized features for cosine similarity
        else:
            return self.l2norm(x_pool), self.l2norm(feat)


# from torchsummary import summary
# model = Network_unimodal(250, arch='resnet50')
# summary(model, [(3, 288, 144),(3, 288, 144)] , batch_size=32)


#   model = Network(250, arch='resnet50')
#   print(resneut50(pretrained= True))
#   print(thermal_module())