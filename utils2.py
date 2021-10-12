import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F


class DCT_trans(nn.Module):
    def __init__(self):
        super(DCT_trans, self).__init__()
        dct_coef = sio.loadmat("./transforms/DCT_coef.mat")['DCT_coef']
        dct_coef = torch.FloatTensor(dct_coef)
        dct_coef = dct_coef.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(data=dct_coef, requires_grad=False)

    def __call__(self, x):
        x = F.conv2d(x, self.weight)
        return x


class IDCT_trans(nn.Module):
    def __init__(self):
        super(IDCT_trans, self).__init__()
        idct_coef = sio.loadmat("./transforms/IDCT_coef.mat")['IDCT_coef']
        idct_coef = torch.FloatTensor(idct_coef)
        idct_coef = idct_coef.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(data=idct_coef, requires_grad=False)

    def __call__(self, x):
        x = F.conv2d(x, self.weight)
        return x


# tf.space_to_depth equivalent
def space_to_depth(x, block_size=8):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)


# tf.depth_to_space equivalent = torch.pixel_shuffle
def depth_to_space(x, block_size=8):
    return nn.functional.pixel_shuffle(x, block_size)


def RGB_to_YUV(cover):
    Y = 0 + 0.299 * cover[:, 0, :, :] + 0.587 * cover[:, 1, :, :] + 0.114 * cover[:, 2, :, :]
    Cb = 0.5 - 0.168736 * cover[:, 0, :, :] - 0.331264 * cover[:, 1, :, :] + 0.5 * cover[:, 2, :, :]
    Cr = 0.5 + 0.5 * cover[:, 0, :, :] - 0.418688 * cover[:, 1, :, :] - 0.081312 * cover[:, 2, :, :]
    return Y.unsqueeze(1),Cb.unsqueeze(1),Cr.unsqueeze(1)

def YUV_to_RGB(Y,Cb,Cr):
    R = Y + 1.402 * (Cr - 0.5)
    G = Y - 0.34414 * (Cb - 0.5) - 0.71414 * (Cr - 0.5)
    B = Y + 1.772 * (Cb - 0.5)
    out = torch.cat([R, G, B], dim=1)
    return out