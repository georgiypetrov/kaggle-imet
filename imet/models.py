from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as M
from .nasnet import nasnetalarge

from .utils import ON_KAGGLE


models_dict = {
    "resnet18": "resnet18/resnet18",
    "resnet34": "pytorch-pretrained-image-models/resnet34",
    "resnet50": "pytorch-pretrained-image-models/resnet50",
    "resnet101": "resnet101/resnet101",
    "resnet152": "resnet152/resnet152",
    "densenet161": "densenet161/densenet161",
    "densenet169": "densenet169/densenet169",
    "densenet121": "pytorch-pretrained-image-models/densenet121",
    "densenet201": "pytorch-pretrained-image-models/densenet201",
    "nasnetalarge": "pytorch-model-zoo/nasnetalarge-a1897284",
    "vgg16": "vgg16/vgg16",
    "vgg19": "vgg19/vgg19",
    "inception_v3": "inceptionv3/inception_v3_google"
}


class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


def create_net(net_cls, pretrained: bool):
    if ON_KAGGLE and pretrained:
        net = net_cls()
        model_name = net_cls.__name__
        model_path = models_dict[model_name]
        weights_path = f'../input/{model_path}.pth'
        print(model_name, weights_path)
        net.load_state_dict(torch.load(weights_path))
    else:
        print("no pretrained or not in kaggle")
        net = net_cls(pretrained=pretrained)
    return net


class ResNet(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=M.resnet50, dropout=False):
        super().__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.net.avgpool = AvgPool()
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        return self.net(x)


class DenseNet(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=M.densenet121):
        super().__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.avg_pool = AvgPool()
        self.net.classifier = nn.Linear(
            self.net.classifier.in_features, num_classes)
        
    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        out = self.net.features(x)
        out = F.relu(out, inplace=True)
        out = self.avg_pool(out).view(out.size(0), -1)
        out = self.net.classifier(out)
        return out


class NasNet(nn.Module):
   def __init__(self, num_classes, pretrained=False, net_cls=nasnetalarge):
       super().__init__()
       self.net = create_net(net_cls, pretrained=pretrained)
       self.avg_pool = AvgPool()
       self.net.last_linear = nn.Linear(
       self.net.last_linear.in_features, num_classes)

   def fresh_params(self):
       return self.net.last_linear.parameters()

   def forward(self, x):
       out = self.net.features(x)
       out = F.relu(out, inplace=True)
       out = self.avg_pool(out).view(out.size(0), -1)
       out = self.net.last_linear(out)
       return out


class VGG(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=M.vgg19):
        super().__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.avg_pool = AvgPool()
        self.net.classifier = nn.Sequential(*([self.net.classifier[i] for i in range(6)] 
        + [nn.Linear(self.net.classifier[6].in_features, num_classes)]))
        
    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        out = self.net.features(x)
        out = F.relu(out, inplace=True)
        out = self.avg_pool(out).view(out.size(0), -1)
        out = self.net.classifier(out)
        return out

    
class InceptionV3(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=M.inception_v3):
        super().__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.net.avgpool = AvgPool()
        self.net.AuxLogits.fc = nn.Linear(self.net.AuxLogits.fc.in_features, num_classes)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        return self.net(x)


resnet18 = partial(ResNet, net_cls=M.resnet18)
resnet34 = partial(ResNet, net_cls=M.resnet34)
resnet50 = partial(ResNet, net_cls=M.resnet50)
resnet101 = partial(ResNet, net_cls=M.resnet101)
resnet152 = partial(ResNet, net_cls=M.resnet152)

densenet121 = partial(DenseNet, net_cls=M.densenet121)
densenet161 = partial(DenseNet, net_cls=M.densenet161)
densenet169 = partial(DenseNet, net_cls=M.densenet169)
densenet201 = partial(DenseNet, net_cls=M.densenet201)

nasnetalarge = partial(NasNet, net_cls=nasnetalarge)

vgg16 = partial(VGG, net_cls=M.vgg16)
vgg19 = partial(VGG, net_cls=M.vgg19)

inception_v3 = partial(InceptionV3, net_cls=M.inception_v3)