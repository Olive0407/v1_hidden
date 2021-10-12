 import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from utils2 import depth_to_space,space_to_depth,YUV_to_RGB,RGB_to_YUV,DCT_trans,IDCT_trans
import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu

class frequency_de(nn.Module):
    def __init__(self,config: HiDDenConfiguration):
        super(frequency_de, self).__init__()
        layers=[]
        self.channels = config.decoder_channels
        self.dct_transform = DCT_trans()
        self.idct = IDCT_trans()
        for _ in range(config.decoder_blocks - 1):     #重复6次
            layers.append(ConvBNRelu(self.channels, self.channels))
        self.layers = nn.Sequential(*layers)

        self.conv=ConvBNRelu(self.channels, config.message_length)
        self.pool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(config.message_length, config.message_length)

    def forward(self, input):
        Y,_,_ = RGB_to_YUV(input)
        out = space_to_depth(Y, 8)
        out = self.dct_transform(out)
        out=self.layers(out)
        out = self.idct(out)
        out=self.conv(out)
        out=self.pool(out)
        out.squeeze_(3).squeeze_(2)
        out= self.linear(out)
        return out