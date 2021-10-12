import torch.nn as nn
import torch
class GaussianNoise(nn.Module):
    def __init__(self,device, std=0.05):
        super(GaussianNoise, self).__init__()
        self.std = std
        self.device=device

    def forward(self, x):
        return self.add_gaussian_noise(x)

    def add_gaussian_noise(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        size = noised_image.size()
        noise = torch.normal(0, torch.ones(size)*self.std)
        noise = noise.to(self.device)
        noised_and_cover[0]=noised_image+noise
        return noised_and_cover