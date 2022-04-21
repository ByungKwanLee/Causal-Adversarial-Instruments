import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Dict, Any, cast
from torch.hub import load_state_dict_from_url
import torch.nn.init as init
from torch.autograd import Variable

class conv_bn_relu(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, padding_size):
        super(conv_bn_relu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=padding_size),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class deconv_bn_relu(nn.Module):
    def __init__(self, in_ch, out_ch, last=False):
        super(deconv_bn_relu, self).__init__()
        if last:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
             )
        else:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2)
            )

    def forward(self, x):
        x = self.deconv(x)
        return x

class CausalIV(nn.Module):
    def __init__(self, ch=False):
        super(CausalIV, self).__init__()

        if ch:
            self.k_list = 3
            self.c_list = [640, 640, 1280, 1280, 640, 640]
            self.nc = 640

        else:
            self.k_list = 3
            self.c_list = [512, 512, 1024, 1024, 512, 512]
            self.nc = 512

        self.encoder = nn.Sequential(
            conv_bn_relu(in_ch=self.nc, out_ch=self.c_list[0], k_size=self.k_list, padding_size=1),
            conv_bn_relu(in_ch=self.c_list[0], out_ch=self.c_list[1], k_size=self.k_list, padding_size=1),
            conv_bn_relu(in_ch=self.c_list[1], out_ch=self.c_list[2], k_size=self.k_list, padding_size=1),
        )
        self.linear = nn.Sequential(nn.Linear(self.c_list[2], self.c_list[3]),
                                    nn.LeakyReLU(0.2))
        self.decoder = nn.Sequential(
            deconv_bn_relu(in_ch=self.c_list[3], out_ch=self.c_list[4]),
            deconv_bn_relu(in_ch=self.c_list[4], out_ch=self.c_list[5]),
            deconv_bn_relu(in_ch=self.c_list[5], out_ch=self.nc, last=True),
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        x = self.encoder(x)
        _, _, h, w = x.shape
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(-1, self.c_list[2])
        x = self.linear(x)
        x = x.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)
        x = self.decoder(x)

        return x

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def exogenous(dataset, ch=False):
    model = CausalIV(ch)

    return model