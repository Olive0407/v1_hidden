import torch.nn as nn
import torch
import  numpy as np
class saltpepper_new(nn.Module):
    def __init__(self, ratio,device):
        super(saltpepper_new, self).__init__()
        self.ratio = ratio
        self.relu = nn.ReLU()
        self.device=device

    def forward(self, noised_and_cover):
        input = noised_and_cover[0]
        size = input[:, 0, :, :].unsqueeze(0).size()
        mask = np.random.choice((-1,0), size=size[1:], p=[self.ratio,  1 - self.ratio])
        # mask = np.repeat(mask, 3, axis=1)
        mask = torch.tensor(mask,dtype=torch.float).to(self.device)

        plus_one = torch.zeros_like(mask)
        for j in range(size[2]):
            for i in range(size[3]):
                if mask[0][j][i] == -1:
                    a_rand = torch.tensor(np.random.choice((0,1)),dtype=torch.float).to(self.device)
                    plus_one[0][j][i] = a_rand

        mask = mask.expand(size)
        plus_one = plus_one.expand(size)
        # input = torch.ceil(input[mask==0])
        # input = torch.trunc(input[mask==1])
        out = self.relu(input+mask) + plus_one
        noised_and_cover[0]=out
        return out