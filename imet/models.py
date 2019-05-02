from torch import nn
from torch.nn import functional as F
from cnn_finetune import make_model


class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


def get_model(model, num_classes, pretrained, input_size, dropout=0.5):
	return make_model(model, num_classes, pretrained, input_size=(input_size, input_size), pool=AvgPool(), dropout_p=dropout)