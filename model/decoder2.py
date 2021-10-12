import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu


class Decoder2(nn.Module):

    def __init__(self, config: HiDDenConfiguration):

        super(Decoder2, self).__init__()
        self.channels = config.decoder_channels  #64

        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(config.decoder_blocks - 4):     #重复6次
            layers.append(ConvBNRelu(self.channels, self.channels))
        layers.append(ConvBNRelu(self.channels, self.channels*2))
        layers.append(ConvBNRelu(self.channels*2, self.channels*4))
        layers.append(ConvBNRelu(self.channels*4, self.channels*8))


        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(512,config.message_length)

    def forward(self, image_with_wm):
        x = self.layers(image_with_wm)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        x.squeeze_(3).squeeze_(2)
        x = self.linear(x)
        return x
