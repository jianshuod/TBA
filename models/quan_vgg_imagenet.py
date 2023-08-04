import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .quantization import *

from typing import Union, List, Dict, Any, Optional, cast

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5, n_bits = 8
    ) -> None:
        super().__init__()
        self.n_bits = n_bits
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            quan_Linear(512 * 7 * 7, 4096, n_bits=n_bits),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            quan_Linear(4096, 4096, n_bits=n_bits),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            quan_Linear(4096, num_classes, n_bits=n_bits),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, quan_Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, quan_Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class VGG_mid(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5, n_bits = 8
    ) -> None:
        super().__init__()
        self.n_bits = n_bits
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            quan_Linear(512 * 7 * 7, 4096, n_bits=n_bits),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            quan_Linear(4096, 4096, n_bits=n_bits),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            quan_Linear(4096, num_classes, n_bits=n_bits),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, quan_Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, quan_Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier[:-1](x)
        return x



def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, n_bits=8) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = quan_Conv2d(in_channels, v, kernel_size=3, padding=1, n_bits=n_bits)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

def vgg16_bn_quan(n_bits):
    return VGG(make_layers(cfgs['D'], True), n_bits=n_bits)

def vgg16_bn_quan_mid(n_bits):
    return VGG_mid(make_layers(cfgs['D'], True), n_bits=n_bits)

def vgg19_bn_quan(n_bits):
    return VGG(make_layers(cfgs['E'], True), n_bits=n_bits)

def vgg19_bn_quan_mid(n_bits):
    return VGG_mid(make_layers(cfgs['E'], True), n_bits=n_bits)