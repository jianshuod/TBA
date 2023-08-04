import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from models.quantization import *


__all__ = ['CifarResNet', 'resnet20_quan', 'resnet32_quan', 'resnet44_quan', 'resnet56_quan', 'resnet110_quan']

def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, quan_Linear) or isinstance(m, quan_Conv2d):
        init.kaiming_normal(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', n_bits=8):
        super(BasicBlock, self).__init__()
        self.conv1 = quan_Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, n_bits=n_bits)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = quan_Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, n_bits=n_bits)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     quan_Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, n_bits=n_bits),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CifarResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, n_bits=8):
        super(CifarResNet, self).__init__()
        self.in_planes = 16
        self.n_bits = n_bits

        self.conv1 = quan_Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, n_bits=self.n_bits)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = quan_Linear(64, num_classes, n_bits=self.n_bits)

        # self.apply(_weights_init)
        # Initialize weights
        for m in self.modules():
            if isinstance(m, quan_Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, n_bits=self.n_bits))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class CifarResNet_mid(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, n_bits=8):
        super(CifarResNet_mid, self).__init__()
        self.in_planes = 16
        self.n_bits = n_bits

        self.conv1 = quan_Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, n_bits=self.n_bits)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = quan_Linear(64, num_classes, n_bits=self.n_bits)

        # self.apply(_weights_init)
        # Initialize weights
        for m in self.modules():
            if isinstance(m, quan_Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, n_bits=self.n_bits))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return out


def resnet20_quan(num_classes=10, n_bits=8):
    model = CifarResNet(BasicBlock, [3, 3, 3], num_classes, n_bits)
    return model

def resnet20_quan_mid(num_classes=10, n_bits=8):
    model = CifarResNet_mid(BasicBlock, [3, 3, 3], num_classes, n_bits)
    return model

def resnet32_quan(num_classes=10, n_bits=8):
    model = CifarResNet(BasicBlock, [5, 5, 5], num_classes, n_bits)
    return model


def resnet44_quan(num_classes=10, n_bits=8):
    model = CifarResNet(BasicBlock, [7, 7, 7], num_classes, n_bits)
    return model


def resnet56_quan(num_classes=10, n_bits=8):
    model = CifarResNet(BasicBlock, [9, 9, 9], num_classes, n_bits)
    return model


def resnet110_quan(num_classes=10, n_bits=8):
    model = CifarResNet(BasicBlock, [18, 18, 18], num_classes, n_bits)
    return model
