import torch
import numpy as np
import torch.nn.functional as F


def total_variation(img):
    # bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.sum(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.sum(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w


def l1_reg(img, constant=0.5):
    return torch.sum(torch.abs(img - constant))


def blur_reg(x, w=3):
    depth = x.shape[1]  # Depth is channel dimension in PyTorch
    k = np.zeros((depth, 1, w, w))
    for ch in range(depth):
        k_ch = k[ch, 0]
        k_ch[:, :] = 0.5
        k_ch[1:-1, 1:-1] = 1.0

    k = torch.from_numpy(k).float().to(x.device)

    def conv_k(t):
        return F.conv2d(t, k, padding=w // 2, groups=depth)

    blurred = conv_k(x) / conv_k(torch.ones_like(x))
    return 0.5 * torch.sum((x - blurred.detach()) ** 2)


def add_regularizer(weight, func):
    return {'weight': weight, 'func': func}