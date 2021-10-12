import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from utils2 import depth_to_space,space_to_depth,YUV_to_RGB,RGB_to_YUV,DCT_trans,IDCT_trans



class RedMark_en(nn.Module):
    def __init__(self, config, strength_factor):
        super(RedMark_en, self).__init__()
        image_size = config.H
        self.block_size = 8
        self.strength_factor = strength_factor
        self.columns = image_size // 8
        self.rows = image_size // 8

        depth = 8 * 8
        self.dct_transform = DCT_trans()
        self.idct = IDCT_trans()
        self.conv1 = nn.Conv2d(depth + 1, depth, 1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(depth, depth, 2, stride=1, padding=0)
        self.conv3 = nn.Conv2d(depth, depth, 2, stride=1, padding=0)
        self.conv4 = nn.Conv2d(depth, depth, 2, stride=1, padding=0)
        self.conv5 = nn.Conv2d(depth, depth, 2, stride=1, padding=0)
        self.activate = nn.ELU()


    def forward(self, cover, watermark):
        Y,Cb,Cr = RGB_to_YUV(cover)

        watermark_size = watermark.size()
        if watermark_size[1]!=1:
            watermark = torch.reshape(watermark,(-1,1,self.columns,self.rows))

        rerange_img = space_to_depth(Y, 8)
        out = self.dct_transform(rerange_img)
        out = torch.cat([out, watermark],dim = 1)
        out = self.activate(self.conv1(out))
        out = self.activate(self.conv2(out))
        out = F.pad(out,[0,1,0,1],mode="replicate")
        out = self.activate(self.conv3(out))
        out = F.pad(out,[0,1,0,1],mode="replicate")
        out = self.activate(self.conv4(out))
        out = F.pad(out,[0,1,0,1],mode="replicate")
        out = self.activate(self.conv5(out))
        out = F.pad(out,[0,1,0,1],mode="replicate")
        out = self.idct(out)
        out = out * self.strength_factor
        out = out + rerange_img
        out = depth_to_space(out)
        return YUV_to_RGB(out,Cb,Cr)


class RedMark_de(nn.Module):
    def __init__(self):
        super(RedMark_de, self).__init__()
        depth = 64
        self.dct_transform = DCT_trans()
        self.conv1 = nn.Conv2d(depth, depth, 1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(depth, depth, 2, stride=1, padding=0)
        self.conv3 = nn.Conv2d(depth, depth, 2, stride=1, padding=0)
        self.conv4 = nn.Conv2d(depth, depth, 2, stride=1, padding=0)
        self.conv5 = nn.Conv2d(depth, 1, 1, stride=1, padding=0)
        self.activate = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        Y,_,_ = RGB_to_YUV(input)
        out = space_to_depth(Y, 8)
        out = self.dct_transform(out)
        out = self.activate(self.conv1(out))
        out = self.activate(self.conv2(out))
        out = F.pad(out,[0,1,0,1],mode="replicate")
        out = self.activate(self.conv3(out))
        out = F.pad(out,[0,1,0,1],mode="replicate")
        out = self.activate(self.conv4(out))
        out = F.pad(out,[0,1,0,1],mode="replicate")
        out = self.sigmoid(self.conv5(out))
        size = out.size()
        out = torch.reshape(out,[-1,size[2]*size[3]])
        return out


class RedMark_en2(nn.Module):
    def __init__(self, config, strength_factor):
        super(RedMark_en2, self).__init__()
        image_size = config.H
        self.block_size = 8
        self.strength_factor = strength_factor
        self.columns = image_size // 8
        self.rows = image_size // 8

        depth = 8 * 8
        self.dct_transform = DCT_trans()
        self.idct = IDCT_trans()

        self.conv1 = nn.Conv2d(depth+1, depth, 1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(depth, depth, 2, stride=1, padding=0)
        self.conv3 = nn.Conv2d(depth, depth, 2, stride=1, padding=0)
        self.conv4 = nn.Conv2d(depth, depth, 2, stride=1, padding=0)
        self.conv5 = nn.Conv2d(depth, depth, 2, stride=1, padding=0)
        self.activate = nn.ELU()
        self.Upsample=torch.nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, cover, watermark):
        Y,Cb,Cr = RGB_to_YUV(cover)

        watermark_size = watermark.size()
        if watermark_size[1]!=1:
            watermark = torch.reshape(watermark, (-1, 1, 8, 8))
            watermark=self.Upsample(watermark)

        rerange_img = space_to_depth(Y, 8)
        out = self.dct_transform(rerange_img)
        out = torch.cat([out, watermark],dim = 1)
        out = self.activate(self.conv1(out))
        out = self.activate(self.conv2(out))
        out = F.pad(out,[0,1,0,1],mode="replicate")
        out = self.activate(self.conv3(out))
        out = F.pad(out,[0,1,0,1],mode="replicate")
        out = self.activate(self.conv4(out))
        out = F.pad(out,[0,1,0,1],mode="replicate")
        out = self.activate(self.conv5(out))
        out = F.pad(out,[0,1,0,1],mode="replicate")
        out = self.idct(out)
        out = out * self.strength_factor
        out = out + rerange_img
        out = depth_to_space(out)
        return YUV_to_RGB(out,Cb,Cr)

class RedMark_de2(nn.Module):
    def __init__(self):
        super(RedMark_de2, self).__init__()
        depth = 64
        self.dct_transform = DCT_trans()
        self.conv1 = nn.Conv2d(depth, depth, 1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(depth, depth, 2, stride=1, padding=0)
        self.conv3 = nn.Conv2d(depth, depth, 2, stride=2, padding=0)
        self.conv4 = nn.Conv2d(depth, depth, 2, stride=1, padding=0)
        self.conv5 = nn.Conv2d(depth, 1, 1, stride=1, padding=0)
        self.activate = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        Y,_,_ = RGB_to_YUV(input)
        out = space_to_depth(Y, 8)
        out = self.dct_transform(out)
        out = self.activate(self.conv1(out))
        out = self.activate(self.conv2(out))
        out = F.pad(out,[0,1,0,1],mode="replicate")
        out = self.activate(self.conv3(out))
        out = self.activate(self.conv4(out))
        out = F.pad(out,[0,1,0,1],mode="replicate")
        out = self.sigmoid(self.conv5(out))
        size = out.size()
        out = torch.reshape(out,[-1,size[2]*size[3]])
        return out

class RedMark_en3(nn.Module):
    def __init__(self, config, strength_factor):
        super(RedMark_en3, self).__init__()
        image_size = config.H
        self.block_size = 8
        self.strength_factor = strength_factor
        self.columns = image_size // 8
        self.rows = image_size // 8

        depth = 8 * 8
        self.dct_transform = DCT_trans()
        self.idct = IDCT_trans()

        self.conv1 = nn.Conv2d(80, depth, 1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(depth, depth, 2, stride=1, padding=0)
        self.conv3 = nn.Conv2d(depth, depth, 2, stride=1, padding=0)
        self.conv4 = nn.Conv2d(depth, depth, 2, stride=1, padding=0)
        self.conv5 = nn.Conv2d(depth, depth, 2, stride=1, padding=0)
        self.activate = nn.ELU()

    def forward(self, cover, watermark):
        Y,Cb,Cr = RGB_to_YUV(cover)

        watermark_size = watermark.size()
        if watermark_size[1]!=1:
            watermark = torch.reshape(watermark, (-1, 1, 8, 8))
            watermark = watermark.repeat(1,16,1,1)

        rerange_img = space_to_depth(Y, 8)
        out = self.dct_transform(rerange_img)
        out = torch.cat([out, watermark],dim = 1)
        out = self.activate(self.conv1(out))
        out = self.activate(self.conv2(out))
        out = F.pad(out,[0,1,0,1],mode="replicate")
        out = self.activate(self.conv3(out))
        out = F.pad(out,[0,1,0,1],mode="replicate")
        out = self.activate(self.conv4(out))
        out = F.pad(out,[0,1,0,1],mode="replicate")
        out = self.activate(self.conv5(out))
        out = F.pad(out,[0,1,0,1],mode="replicate")
        out = self.idct(out)
        out = out * self.strength_factor
        out = out + rerange_img
        out = depth_to_space(out)
        return YUV_to_RGB(out,Cb,Cr)

class RedMark_de3(nn.Module):
    def __init__(self):
        super(RedMark_de3, self).__init__()
        depth = 64
        self.dct_transform = DCT_trans()
        self.conv1 = nn.Conv2d(depth, depth, 1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(depth, depth, 2, stride=1, padding=0)
        self.conv3 = nn.Conv2d(depth, depth, 2, stride=1, padding=0)
        self.conv4 = nn.Conv2d(depth, depth, 2, stride=1, padding=0)
        self.conv5 = nn.Conv2d(depth, 1, 1, stride=1, padding=0)
        self.activate = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        Y,_,_ = RGB_to_YUV(input)
        out = space_to_depth(Y, 8)
        out = self.dct_transform(out)
        out = self.activate(self.conv1(out))
        out = self.activate(self.conv2(out))
        out = F.pad(out,[0,1,0,1],mode="replicate")
        out = self.activate(self.conv3(out))
        out = F.pad(out,[0,1,0,1],mode="replicate")
        out = self.activate(self.conv4(out))
        out = F.pad(out,[0,1,0,1],mode="replicate")
        out = self.sigmoid(self.conv5(out))
        size = out.size()
        out = torch.reshape(out,[-1,size[2]*size[3]])
        return out