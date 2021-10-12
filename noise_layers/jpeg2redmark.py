import torch
import torch.nn as nn


Q_matrix = torch.tensor([16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99])

class redmarkjpeg(nn.Module):
    def __init__(self, device,Q=80):
        super(redmarkjpeg, self).__init__()
        self.device=device
        self.Q=Q
        if  Q < 50:
            S = 5000 / Q
        else:
            S = 200 - 2 * Q
        Ts = ((S * Q_matrix + 50) / 100).floor()
        Ts[Ts == 0] = 1
        self.Q_matrix=Ts

    def add_gaussian_noise(self, x, device):
        size = x.size()
        noise = torch.normal(0,torch.ones(size))
        noise = noise.to(device)
        return x + noise

    def forward(self, noised_and_cover):

        noised_image = noised_and_cover[0]
        x = noised_image
        Q_flatten = self.Q_matrix.to(self.device).float()
        a = x.size()[2]
        b = x.size()[3]
        c = x.size()[0]
        for n in range(c):
            for i in range(a):
                for j in range(b):
                    x[n, :, i, j] = ((x[n, :, i, j] * 255) / Q_flatten)
        y = self.add_gaussian_noise(noised_image,self.device)
        for n in range(c):
            for i in range(a):
                for j in range(b):
                    y[n, :, i, j] = (y[n, :, i, j] * Q_flatten / 255)
        noised_and_cover[0]=y
        return noised_and_cover


