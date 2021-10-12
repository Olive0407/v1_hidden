import torch.nn as nn
from model.encoder import Encoder
from model.decoder2 import Decoder2
from model.decoder import Decoder
from model.decoder3 import Decoder3
from model.frequency_decoder import frequency_de
from model.frequency_encoder import frequency_en2,frequency_en
from options import HiDDenConfiguration
from noise_layers.noiser import Noiser
from redmark import RedMark_en,RedMark_de,RedMark_en2,RedMark_de2,RedMark_en3,RedMark_de3

class EncoderDecoder(nn.Module):
    """
    Combines Encoder->Noiser->Decoder into single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a three-tuple: (encoded_image, noised_image, decoded_message)
    """
    def __init__(self, config: HiDDenConfiguration, noiser: Noiser, redmark: bool = False):

        super(EncoderDecoder, self).__init__()
        if redmark:
            self.encoder = RedMark_en3(config,1)
            self.decoder = RedMark_de3()
        else:
            self.encoder = frequency_en(config)
            self.decoder = frequency_de(config)


        self.noiser = noiser



    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        noised_and_cover = self.noiser([encoded_image, image])   #返回的是一个长度为2的列表，其中0为攻击后的图片，1为原始的载体图像
        noised_image = noised_and_cover[0]
        decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message
