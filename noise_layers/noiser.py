import numpy as np
import torch.nn as nn
from noise_layers.identity import Identity
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.quantization import Quantization
from noise_layers.GaussianNoise import GaussianNoise
from noise_layers.sharpen import sharpen
from noise_layers.saltpepper import saltpepper
from noise_layers.GaussianBlurConv import GaussianBlurConv
from noise_layers.Block_Drop_out import Block_Drop_out
from noise_layers.saltpepper_new import saltpepper_new
from noise_layers.redmarkjpeg import redmarkjpeg
class Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self, noise_layers: list, device):
        super(Noiser, self).__init__()
        #self.noise_layers = [Identity()]    #identity不做攻击
        self.noise_layers = []
        for layer in noise_layers:
            if type(layer) is str:
                if layer == 'JpegPlaceholder':
                    self.noise_layers.append(JpegCompression(device))   #jpeg攻击
                elif layer == 'QuantizationPlaceholder':
                    self.noise_layers.append(Quantization(device))
                elif layer == 'GaussianNoise':
                    self.noise_layers.append(GaussianNoise(device))
                elif layer == 'sharpen':
                    self.noise_layers.append(sharpen(device))
                elif layer == 'saltpepper':
                    self.noise_layers.append(saltpepper())
                elif layer == 'Block_Drop_out':
                    self.noise_layers.append(Block_Drop_out(device))
                elif layer == 'GaussianBlurConv':
                    self.noise_layers.append(GaussianBlurConv(device))
                elif layer == 'redmarkjpeg':
                    self.noise_layers.append(redmarkjpeg(device))
                else:
                    raise ValueError(f'Wrong layer placeholder string in Noiser.__init__().'
                                     f' Expected "JpegPlaceholder" or "QuantizationPlaceholder" but got {layer} instead')
            else:
                self.noise_layers.append(layer)
        # self.noise_layers = nn.Sequential(*noise_layers)

    def forward(self, encoded_and_cover):
        random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
        back=random_noise_layer(encoded_and_cover)
        return back

