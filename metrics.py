import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size / 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window

def SSIM(img1, img2):
    (_, channel, _, _) = img1.size()
    window_size = 11
    window = create_window(window_size, channel).cuda()
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def PSNR(img1, img2, mask=None):
    if mask is not None:
        mse = (img1 - img2) ** 2
        B, C, H, W = mse.size()
        mse = torch.sum(mse * mask.float()) / (torch.sum(mask.float()) * C)
    else:
        mse = torch.mean((img1 - img2) ** 2)

    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def RMSE(img1, img2, mask=None):
    if mask is not None:
        mse = (img1 - img2) ** 2
        B, C, H, W = mse.size()
        mse = torch.sum(mse * mask.float()) / (torch.sum(mask.float()) * C)
    else:
        mse = torch.mean((img1 - img2) ** 2)
    return math.sqrt(mse)

# Spectral-Angle-Mapper (SAM)
def SAM(H_fuse, H_ref):
    # Compute number of spectral bands
    N_spectral = H_fuse.shape[1]

    # Rehsaping fused and reference data
    H_fuse_reshaped = H_fuse.view(N_spectral, -1)
    H_ref_reshaped = H_ref.view(N_spectral, -1)
    N_pixels = H_fuse_reshaped.shape[1]

    # Calculating inner product
    inner_prod = torch.nansum(H_fuse_reshaped * H_ref_reshaped, 0)
    fuse_norm = torch.nansum(H_fuse_reshaped ** 2, dim=0).sqrt()
    ref_norm = torch.nansum(H_ref_reshaped ** 2, dim=0).sqrt()

    # Calculating SAM
    SAM = torch.rad2deg(torch.nansum(torch.acos(inner_prod / (fuse_norm * ref_norm))) / N_pixels)
    return SAM



# Erreur Relative Globale Adimensionnelle De Synthèse (ERGAS)
def ERGAS(H_fuse, H_ref):
    # Compute number of spectral bands
    N_spectral = H_fuse.shape[1]

    # Reshaping images
    H_fuse_reshaped = H_fuse.view(N_spectral, -1)
    H_ref_reshaped = H_ref.view(N_spectral, -1)
    N_pixels = H_fuse_reshaped.shape[1]

    # Calculating RMSE of each band
    rmse = torch.sqrt(torch.nansum((H_ref_reshaped - H_fuse_reshaped) ** 2, dim=1) / N_pixels)
    mu_ref = torch.mean(H_ref_reshaped, dim=1)

    # Calculating Erreur Relative Globale Adimensionnelle De Synthèse (ERGAS)
    ERGAS = 100 * (1 / 4 ** 2) * torch.sqrt(torch.nansum(torch.div(rmse, mu_ref) ** 2) / N_spectral)
    return ERGAS



