import torch
import torch.nn as nn
import torch.nn.functional as F

from models.quantization import *

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class FakeReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class VGG(nn.Module):
    def __init__(self, vgg_name, n_bits):
        super(VGG, self).__init__()
        self.n_bits = n_bits
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = quan_Linear(512, 10, n_bits=n_bits)

    def forward(self, x, with_latent=False, fake_relu=False):
        out = self.features[0:-3](x)
        if fake_relu:
            out = FakeReLU.apply(out)
        else: out = self.features[-3](out)
        out = self.features[-2:](out)
        out = out.view(out.size(0), -1)
        final = self.classifier(out)
        if with_latent:
            return final, out
        else:return final


    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [quan_Conv2d(in_channels, x, kernel_size=3, padding=1, n_bits=self.n_bits),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class VGG_mid(nn.Module):
    def __init__(self, vgg_name, n_bits):
        super(VGG_mid, self).__init__()
        self.n_bits = n_bits
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = quan_Linear(512, 10, n_bits=n_bits)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [quan_Conv2d(in_channels, x, kernel_size=3, padding=1, n_bits=self.n_bits),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def vgg16_quan(bit_width):
    return VGG("VGG16", bit_width)

def vgg16_quan_mid(bit_width):
    return VGG_mid("VGG16", bit_width)

def vgg19_quan(bit_width):
    return VGG("VGG19", bit_width)

def vgg19_quan_mid(bit_width):
    return VGG_mid("VGG19", bit_width)