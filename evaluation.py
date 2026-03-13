# -*- coding: utf-8 -*-
"""
functions used to compute the metrics
"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def compute_sam(x_true, x_pred):
    assert x_true.ndim == 3 and x_true.shape == x_pred.shape
    w, h, c = x_true.shape
    x_true = x_true.reshape(-1, c)
    x_pred = x_pred.reshape(-1, c)
    sam = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1) + 1e-7)
    sam = np.arccos(sam) * 180 / np.pi
    mSAM = sam.mean()

    return mSAM


def compute_psnr(x_true, x_pred):
    assert x_true.ndim == 3 and x_pred.ndim == 3
    img_w, img_h, img_c = x_true.shape
    ref = x_true.reshape(-1, img_c)
    tar = x_pred.reshape(-1, img_c)
    msr = np.mean((ref - tar) ** 2, 0)
    max2 = np.max(ref, 0) ** 2
    psnrall = 10 * np.log10(max2 / msr)
    m_psnr = np.mean(psnrall)
    psnr_all = psnrall.reshape(img_c)
    return m_psnr


def compare_mpsnr(x_true, x_pred, data_range):
    """
    :param x_true: Input image must have three dimension (H, W, C)
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    channels = x_true.shape[2]
    total_psnr = [compare_psnr(image_true=x_true[:, :, k], image_test=x_pred[:, :, k], data_range=data_range)
                  for k in range(channels)]

    return np.mean(total_psnr)

def compute_ergas(x_true, x_pred, scale_factor):
    assert x_true.ndim == 3 and x_pred.ndim == 3 and x_true.shape == x_pred.shape
    img_w, img_h, img_c = x_true.shape
    err = x_true - x_pred
    ERGAS = 0
    for i in range(img_c):
        ERGAS = ERGAS + np.mean(err[:, :, i] ** 2 / np.mean(x_true[:, :, i]) ** 2)
    ERGAS = (100 / scale_factor) * np.sqrt((1 / img_c) * ERGAS)
    return ERGAS


def compute_cc(x_true, x_pred):
    img_w, img_h, img_c = x_true.shape
    result = np.ones((img_c,))
    for i in range(0, img_c):
        CCi = np.corrcoef(x_true[:, :, i].flatten(), x_pred[:, :, i].flatten())
        result[i] = CCi[0, 1]
    return result.mean()


def compute_rmse(x_true, x_pre):
    img_w, img_h, img_c = x_true.shape
    return np.sqrt(((x_true - x_pre) ** 2).sum() / (img_w * img_h * img_c))




def compare_mssim(x_true, x_pred, data_range, multidimension= False):
    """

    :param x_true:
    :param x_pred:
    :param data_range:
    :param multidimension:
    :return:
    """
    mssim = [compare_ssim(im1=x_true[:, :, i], im2=x_pred[:, :, i], data_range=data_range, multidimension=multidimension)
            for i in range(x_true.shape[2])]

    return np.mean(mssim)