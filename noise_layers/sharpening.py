import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
class sharpen(nn.Module):
    def __init__(self,device, channels=3):
        super(sharpen, self).__init__()
        self.channels = channels
        self.device=device
        kernel = [[0,-1,0],
                  [-1,5,-1],
                  [0,-1,0]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).to(device)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, noised_and_cover):
        x=noised_and_cover[0]
        x = F.conv2d(x, self.weight, padding=2, groups=self.channels)
        noised_and_cover[0]=x
        return noised_and_cover