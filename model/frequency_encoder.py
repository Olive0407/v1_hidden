import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from utils2 import depth_to_space,space_to_depth,YUV_to_RGB,RGB_to_YUV,DCT_trans,IDCT_trans
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu


class  frequency_en(nn.Module):
    def __init__(self, config):
        super(frequency_en, self).__init__()
        image_size = config.H
        self.block_size = 8
        self.columns = image_size // 8
        self.rows = image_size // 8
        self.H = config.H  #128*128
        self.W = config.W
        self.conv_channels = config.encoder_channels  #64
        self.num_blocks = config.encoder_blocks  #4

        self.dct_transform = DCT_trans()
        self.idct = IDCT_trans()
        cover_layers = []
        for _ in range(3):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            cover_layers.append(layer)
        self.conv_layers1 = nn.Sequential(* cover_layers)
        cancat_layers=[]
        cancat_layers.append(ConvBNRelu(self.conv_channels+config.message_length, self.conv_channels))
        for _ in range(2):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            cancat_layers.append(layer)
        self.conv_layers2 = nn.Sequential(*cancat_layers)

        layer=ConvBNRelu(1, 3)
        layer2=ConvBNRelu(3, 1)
        self.conv_layers3=nn.Sequential(layer,layer2)


    def forward(self, cover, watermark):
        Y,Cb,Cr = RGB_to_YUV(cover)

        rerange_img = space_to_depth(Y, 8)
        out = self.dct_transform(rerange_img)
        out=self.conv_layers1(out)
        expanded_message =watermark.unsqueeze(-1)    #8 30 1
        expanded_message.unsqueeze_(-1)  # 8 30 1 1
        watermark = expanded_message.expand(-1,-1,  self.columns,  self.rows)
        out = torch.cat([out, watermark],dim = 1)
        out=self.conv_layers2(out)
        out = self.idct(out)
        out = depth_to_space(out)
        out=self.conv_layers3(out)
        return YUV_to_RGB(out,Cb,Cr)




class  frequency_en2(nn.Module):
    def __init__(self, config):
        super(frequency_en2, self).__init__()
        image_size = config.H
        self.block_size = 8
        self.columns = image_size // 8
        self.rows = image_size // 8
        self.H = config.H  #128*128
        self.W = config.W
        self.conv_channels = config.encoder_channels  #64
        self.num_blocks = config.encoder_blocks  #4

        self.dct_transform = DCT_trans()
        self.idct = IDCT_trans()
        #cover_layers = []
        #for _ in range(3):
        #layer = ConvBNRelu(self.conv_channels, self.conv_channels)
        #cover_layers.append(layer)
        #self.conv_layers1 = nn.Sequential(* cover_layers)
        cancat_layers=[]
        cancat_layers.append(ConvBNRelu(self.conv_channels+16, self.conv_channels))
        for _ in range(2):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            cancat_layers.append(layer)
        self.conv_layers2 = nn.Sequential(*cancat_layers)
        self.Upsample=torch.nn.Upsample(scale_factor=2, mode='nearest')


    def forward(self, cover, watermark):
        Y,Cb,Cr = RGB_to_YUV(cover)

        rerange_img = space_to_depth(Y, 8)
        out = self.dct_transform(rerange_img)
        #out=self.conv_layers1(out)
        #expanded_message =watermark.unsqueeze(-1)    #8 30 1
        #expanded_message.unsqueeze_(-1)  # 8 30 1 1
        watermark_size = watermark.size()

        if watermark_size[1]!=1:
            temp=int(watermark_size[1]**0.5)
            watermark = torch.reshape(watermark,(-1,1,temp,temp))
        watermark=self.Upsample(watermark)
        watermark = watermark.repeat(1,16,1,1)
        out = torch.cat([out, watermark],dim = 1)
        out=self.conv_layers2(out)
        out = self.idct(out)
        out = depth_to_space(out)
        return YUV_to_RGB(out,Cb,Cr)