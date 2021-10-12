import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class GaussianBlurConv(nn.Module):
    def __init__(self, device,channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        self.device=device
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0).to(device)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, noised_and_cover):
        x = noised_and_cover[0]
        noised_and_cover[0]=F.conv2d(x, self.weight, padding=2, groups=self.channels)
        return noised_and_cover