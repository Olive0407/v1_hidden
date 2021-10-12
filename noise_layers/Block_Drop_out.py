import torch.nn as nn
import torch
import  numpy as np

class Block_Drop_out(nn.Module):
    # this is a module to dropout some blocks of 8*8
    # when we get the space_to_depth image, randomly drop a unit, it means the whole block is gone.
    def __init__(self,device,ratio=0.3):
        super(Block_Drop_out, self).__init__()
        self.ratio = ratio
        self.relu = nn.ReLU()
        self.device=device

    def forward(self, noised_and_cover):

        input=noised_and_cover[0]
        size = input[:,0,:,:].unsqueeze(1).size()
        mask = np.random.choice([-1, 0], size, p=[self.ratio,1-self.ratio])
        mask = torch.tensor(mask,dtype=torch.float).to(self.device)
        mask = mask.expand(input.size())
        out = self.relu(input+mask)
        noised_and_cover[0]=out
        # tmp = torch.rand(size)
        # tmp = tmp > self.ratio
        # tmp = np.repeat(tmp,3,axis=1)
        # tmp = tmp.type(torch.FloatTensor)
        # tmp[tmp<0.5] += 0.0005
        return noised_and_cover