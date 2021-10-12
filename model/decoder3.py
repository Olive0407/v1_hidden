import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu


class Decoder3(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, config: HiDDenConfiguration):

        super(Decoder3, self).__init__()
        self.channels = config.decoder_channels  #64

        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(config.decoder_blocks - 1):     #重复6次
            layers.append(ConvBNRelu(self.channels, self.channels))

        layers.append(ConvBNRelu(self.channels, config.message_length))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.final=nn.Conv2d(config.message_length, config.message_length, kernel_size=1, stride=1, padding=0)
        self.layers = nn.Sequential(*layers)


    def forward(self, image_with_wm):
        x = self.layers(image_with_wm)
        x=self.final(x)
        x.squeeze_(3).squeeze_(2)
        return x
