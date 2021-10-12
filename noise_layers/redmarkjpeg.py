import torch
import torch.nn as nn
import scipy.io as sio
import torch.nn.functional as F

Tb = torch.tensor([16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99])
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

class DCT_trans(nn.Module):
    def __init__(self):
        super(DCT_trans, self).__init__()
        dct_coef = sio.loadmat("./transforms/DCT_coef.mat")['DCT_coef']
        dct_coef = torch.FloatTensor(dct_coef)
        dct_coef = dct_coef.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(data=dct_coef, requires_grad=False)

    def __call__(self, x):
        device=x.device
        x = F.conv2d(x, (self.weight).to(device))
        return x


class IDCT_trans(nn.Module):
    def __init__(self):
        super(IDCT_trans, self).__init__()
        idct_coef = sio.loadmat("./transforms/IDCT_coef.mat")['IDCT_coef']
        idct_coef = torch.FloatTensor(idct_coef)
        idct_coef = idct_coef.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(data=idct_coef, requires_grad=False)

    def __call__(self, x):
        device=x.device
        x = F.conv2d(x, (self.weight).to(device))
        return x

class redmarkjpeg(nn.Module):
    def __init__(self,device,Q=30):
        super(redmarkjpeg, self).__init__()
        self.device=device
        if  Q < 50:
            S = 5000 / Q
        else:
            S = 200 - 2 * Q
        Ts = ((S * Tb + 50) / 100).float().floor()
        Ts[Ts == 0] = 1
        Ts=Ts.unsqueeze(-1)
        Ts = Ts.unsqueeze(-1)
        Ts = Ts.unsqueeze(0)
        self.Q_flatten=Ts
        self.dct = DCT_trans()
        self.idct = IDCT_trans()

    def add_gaussian_noise(self, x, device):
        size = x.size()
        noise = torch.normal(0,torch.ones(size))
        noise = noise.to(device)
        return x + noise

    def forward(self, noised_and_cover):

        noised_image = noised_and_cover[0]
        x, Cb, Cr = RGB_to_YUV(noised_image)
        x = space_to_depth(x, 8)
        x=self.dct(x)
        a=x.size()[0]
        b=x.size()[2]
        c=x.size()[3]
        Q_matrix = self.Q_flatten.repeat(a, 1, b, c).to(self.device)
        x_new=torch.div((x*255),Q_matrix)
        y = self.add_gaussian_noise(x_new,self.device)
        y_new = torch.zeros(y.size()).to(self.device)
        y_new=(y_new*Q_matrix)/255
        y_new = self.idct(y_new)
        y_new = depth_to_space(y_new, 8)
        noised_and_cover[0]=YUV_to_RGB(y_new, Cb, Cr)

        return noised_and_cover


