import torch.nn as nn
import torch


class saltpepper(nn.Module):
    def __init__(self, ratio=0.04):
        super(saltpepper, self).__init__()
        self.ratio = ratio
        #self.device=device

    def forward(self, noised_and_cover):
        input=noised_and_cover[0]
        size = input.size()
        rand_num = torch.rand(size)
        input[rand_num < self.ratio / 2] = 1
        input[rand_num > 1 - self.ratio / 2] = 0
        noised_and_cover[0]=input
        return noised_and_cover