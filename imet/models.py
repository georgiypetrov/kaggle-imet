from torch import nn
from torch.nn import functional as F
from cnn_finetune import make_model


class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


def se_resnet50(num_classes, pretrained=False):
    return make_model('se_resnet50', num_classes, pretrained, input_size=(288, 288), pool=AvgPool())


def resnet18(num_classes, pretrained=False):
    return make_model('resnet18', num_classes, pretrained, input_size=(288, 288), pool=AvgPool())


def resnet34(num_classes, pretrained=False):
    return make_model('resnet34', num_classes, pretrained, input_size=(288, 288), pool=AvgPool())


def resnet50(num_classes, pretrained=False):
    return make_model('resnet50', num_classes, pretrained, input_size=(288, 288), pool=AvgPool())


def resnet101(num_classes, pretrained=False):
    return make_model('resnet101', num_classes, pretrained, input_size=(288, 288), pool=AvgPool())


def nasnetalarge(num_classes, pretrained=False):
    return make_model('nasnetalarge', num_classes, pretrained, input_size=(288, 288), pool=AvgPool())


def densenet121(num_classes, pretrained=False):
    return make_model('densenet121', num_classes, pretrained, input_size=(288, 288), pool=AvgPool())


def densenet201(num_classes, pretrained=False):
    return make_model('densenet201', num_classes, pretrained, input_size=(288, 288), pool=AvgPool())


def mobilenet(num_classes, pretrained=False):
    return make_model('mobilenet', num_classes, pretrained, input_size=(288, 288), pool=AvgPool())    
