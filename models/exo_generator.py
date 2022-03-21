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
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_ch),
             )
        else:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2)
            )

    def forward(self, x):
        x = self.deconv(x)
        return x

class CausalIV(nn.Module):
    def __init__(self, dataset, z_dim=10, nc=512):
        super(CausalIV, self).__init__()
        self.z_dim = z_dim
        self.nc = nc

        if dataset == 'imagenet':
            self.c_list = []
            self.k_list = []
        else:
            self.k_list = 3
            self.c_list = [512, 512, 1024, 1024, 512, 512]

        self.encoder = nn.Sequential(
            conv_bn_relu(in_ch=nc, out_ch=self.c_list[0], k_size=self.k_list, padding_size=1),
            conv_bn_relu(in_ch=self.c_list[0], out_ch=self.c_list[1], k_size=self.k_list, padding_size=1),
            conv_bn_relu(in_ch=self.c_list[1], out_ch=self.c_list[2], k_size=self.k_list, padding_size=1),
        )
        self.linear = nn.Sequential(nn.Linear(self.c_list[2], self.c_list[2]),
                                    nn.LeakyReLU(0.2))
        self.decoder = nn.Sequential(
            deconv_bn_relu(in_ch=self.c_list[2], out_ch=self.c_list[3]),
            deconv_bn_relu(in_ch=self.c_list[3], out_ch=self.c_list[4]),
            deconv_bn_relu(in_ch=self.c_list[4], out_ch=self.c_list[5], last=True),
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        _, _, h, w = x.shape
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

class ExoNet(nn.Module):

    def __init__(self, features: nn.Module, init_weights: bool = True,
                 mean: torch.tensor = None, std: torch.tensor = None) -> None:
        super(ExoNet, self).__init__()

        # configuration
        self.mean = mean.view(1, -1, 1, 1)
        self.std = std.view(1, -1, 1, 1)

        self.features = features

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std # normalize by LBK
        x = self.features(x)

        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'L':
            v = 512
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v)]
            else:
                layers += [conv2d]
            in_channels = v
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs: Dict[int, List[Union[str, int]]] = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 'L'],
}

def exogenous(dataset, mean, std, exo=False, network=None):
    if exo:
        model = CausalIV(dataset)
    else:
        model = ExoNet(features=make_layers(cfgs[11], batch_norm=True), mean=mean, std=std)

    return model