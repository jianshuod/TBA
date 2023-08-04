import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from bitstring import Bits


class augLag_Base(nn.Module):
    def __init__(self, n_bits, w, b, step_size, init=False):
        super(augLag_Base, self).__init__()
        self.n_bits = n_bits
        self.b = nn.Parameter(torch.tensor(b).float(), requires_grad=True)
        self.w_twos = nn.Parameter(torch.zeros(list(w.shape) + [self.n_bits]),
                                   requires_grad=True)
        self.step_size = step_size
        self.w = w

        base = [2 ** i for i in range(self.n_bits - 1, -1, -1)]
        base[0] = -base[0]
        self.base = nn.Parameter(torch.tensor([[base]]).float())

        if init:
            self.reset_w_twos()

    def reset_w_twos(self):
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                self.w_twos.data[i][j] += \
                    torch.tensor([int(b) for b in Bits(int=int(self.w[i][j]),
                                                       length=self.n_bits).bin])


class augLag_Linear(augLag_Base):

    def __init__(self, n_bits, w, b, step_size, init=False):
        super(augLag_Linear, self).__init__(n_bits, w, b, step_size, init)

    def forward(self, x):
        # covert w_twos to float
        w = self.w_twos * self.base
        w = torch.sum(w,
                      dim=2) * self.step_size  # step size is the delta in quantization process. this line corresponds to the dequantization process

        # calculate output
        x = F.linear(x, w, self.b)

        return x


class augLag_Conv2(augLag_Base):
    def __init__(self, n_bits, w, b, step_size, stride, padding, dilation,
                 groups, init=False):
        super(augLag_Conv2, self).__init__(n_bits, w, b, step_size, init)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):

        # covert w_twos to float
        w = self.w_twos * self.base
        w = torch.sum(w,
                      dim=4) * self.step_size  # step size is the delta in quantization process. this line corresponds to the dequantization process

        # calculate output
        x = F.conv2d(x, w, self.b, self.stride,
                     self.padding, self.dilation, self.groups)

        return x

    def reset_w_twos(self):
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                for k in range(self.w.shape[2]):
                    for z in range(self.w.shape[3]):
                        self.w_twos.data[i][j][k][z] += \
                            torch.tensor([int(b) for b in
                                          Bits(int=int(self.w[i][j][k][z]),
                                               length=self.n_bits).bin])