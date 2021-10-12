'''
#### this model.py is for the fail 16 in 3 or 1
made by audience on Dec 4th,2018
'''
from abc import ABC

import torch
import torchvision
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt


# Inception Block

class Inception(nn.Module):
    def __init__(self, in_channel, out_channel, batch_norm=True):
        super(Inception, self).__init__()
        # branch1  1*1 conv
        self.conv1x1_1 = nn.Conv2d(in_channel, out_channel // 4, 1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel // 4)

        # branch2  1*1+3*3
        self.conv1x1_2 = nn.Conv2d(in_channel, out_channel // 4, 1, stride=1, padding=0, bias=False)
        self.conv3x3_2 = nn.Conv2d(out_channel // 4, out_channel // 4, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel // 4)
        # branch3  1*1+5*5
        self.conv1x1_3 = nn.Conv2d(in_channel, out_channel // 4, 1, stride=1, padding=0, bias=False)
        self.conv5x5_3 = nn.Conv2d(out_channel // 4, out_channel // 4, 5, stride=1, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel // 4)
        # branch4  3*3 max+1*1
        self.maxpooling3x3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv1x1_4 = nn.Conv2d(in_channel, out_channel // 4, 1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channel // 4)
        # branch5  identity
        # other function
        self.relu = nn.LeakyReLU()
        self.identity = nn.Conv2d(in_channel, out_channel, 1, stride=1, padding=0, bias=False)

    def forward(self, input):
        branch1 = self.relu(self.bn1(self.conv1x1_1(input)))

        out = self.conv3x3_2(self.conv1x1_2(input))
        branch2 = self.relu(self.bn2(out))

        out = self.conv5x5_3(self.conv1x1_3(input))
        branch3 = self.relu(self.bn3(out))

        out = self.conv1x1_4(self.maxpooling3x3(input))
        branch4 = self.relu(self.bn4(out))

        identity = self.identity(input)

        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        output = self.relu(out + identity)
        return output


class Encoder(nn.Module):
    def __init__(self, in_channel=2, out_channel=1):
        super(Encoder, self).__init__()
        self.in_channel = in_channel
        # convblock1
        self.conv1 = nn.Conv2d(self.in_channel, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Inception series
        self.inception = nn.Sequential(
            Inception(16, 32),
            Inception(32, 64),
            Inception(64, 128),
            Inception(128, 256),
            Inception(256, 128),
            Inception(128, 64),
            Inception(64, 32),

        )

        # convblocck2
        self.conv2 = nn.Conv2d(32, 16, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)

        # convblock3
        if in_channel == 2:
            self.conv3 = nn.Conv2d(16, 1, 1, stride=1, padding=0)
        else:
            self.conv3 = nn.Conv2d(16, 3, 1, stride=1, padding=0)

        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, cover, secret):

        # RGB to YCrCb
        Y = 0 + 0.299 * cover[:, 0, :, :] + 0.587 * cover[:, 1, :, :] + 0.114 * cover[:, 2, :, :]
        Cb = 0.5 - 0.168736 * cover[:, 0, :, :] - 0.331264 * cover[:, 1, :, :] + 0.5 * cover[:, 2, :, :]
        Cr = 0.5 + 0.5 * cover[:, 0, :, :] - 0.418688 * cover[:, 1, :, :] - 0.081312 * cover[:, 2, :, :]
        Y = Y.view(-1, 1, 256, 256)
        Cb = Cb.view(-1, 1, 256, 256)
        Cr = Cr.view(-1, 1, 256, 256)

        if self.in_channel == 2:
            input_ = torch.cat([Y, secret], dim=1)
        if self.in_channel == 4:
            input_ = torch.cat([cover, secret], dim=1)

        layer1 = self.relu(self.bn1(self.conv1(input_)))
        inception_out = self.inception(layer1)
        layer2 = self.relu(self.bn2(self.conv2(inception_out)))
        # output=self.tanh(self.conv3(layer2))
        # the paper say tanh, but i think the value should range(0,1) so i wanna use sigmoid
        output = self.sigmoid(self.conv3(layer2))

        if self.in_channel == 2:
            output = output.view(-1, 1, 256, 256)

            # reconstruct the 3 channel pic

            # stego0 = output + 1.402 * (Cr - 128)
            # stego1 = output - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
            # stego2 = output + 1.772 * (Cb - 128)
            # stego = torch.cat([stego0, stego1, stego2], dim=1)
            stego = torch.cat([output, Cb, Cr], dim=1)

        else:
            output = output.view(-1, 3, 256, 256)
            stego = output

        return stego

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.xavier_uniform(m.weight)

#hidden
class Decoder2d(nn.Module):
    def __init__(self, config, intent = False):
        super(Decoder2d, self).__init__()
        self.dropout = True
        self.intent = intent
        self.in_channels = 3
        self.out_channels = config.message_length
        self.conv_1 = nn.Conv2d(self.in_channels, 16, 3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(16, 64, 3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.3)
        self.incep1 = Inception(64, 128, False)

        if intent:
            self.conv_3 = nn.Conv2d(128, 512, 3, stride=1, padding=1)
            self.adp_pool = nn.AdaptiveAvgPool2d(1)
            self.linear = nn.Linear(512, self.out_channels)
        else:
            self.adp_pool = nn.AdaptiveAvgPool2d(21)
            self.conv3 = nn.Conv2d(128,64,3,1,1)
            self.conv4 = nn.Conv2d(64,16,3,1,1)
            self.conv5 = nn.Conv2d(16,1,1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()

    def forward(self, input_):


        out = self.relu(self.conv_1(input_))
        out = self.relu(self.conv_2(out))
        out = self.dropout(out)
        out = self.incep1(out)

        if self.intent:
            out = self.relu(self.conv_3(out))
            out = self.adp_pool(out)
            out.squeeze_(3).squeeze_(2)
            out = self.linear(out)
            out = self.sigmoid(out)
        else:
            out = self.adp_pool(out)
            out = self.relu(self.conv3(out))
            out = self.relu(self.conv4(out))
            out = self.sigmoid(self.conv5(out))

        return out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.xavier_uniform(m.weight)


