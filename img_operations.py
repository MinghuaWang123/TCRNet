import rasterio
import os
import torch
import cv2
import numpy as np
import numpy as np
import torch as t
import math
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
import random
from lpips.lpips import LPIPS
from pytorch_msssim import msssim, ssim
import torchvision
import torch.nn as nn
import torchvision.transforms.functional as TF

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def spatial_edge(x):
    edge1 = x[:, :, 0:x.size(2)-1, :] - x[:, :, 1:x.size(2), :]
    edge2 = x[:, :, :, 0:x.size(3)-1] - x[:, :,  :, 1:x.size(3)]

    return edge1, edge2

def spectral_edge(x):
    edge = x[:, 0:x.size(1)-1, :, :] - x[:, 1:x.size(1), :, :]

    return edge

def hyperRMSE(self, S_est, sor):
    N = np.size(self.S, 0)
    S_est = S_est[:, sor]
    rmse = self.S - S_est
    rmse = rmse * rmse;
    rmse = np.mean(np.sqrt(np.sum(rmse, 0) / N))
    return rmse


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


#保存网络
def save_state_dict(net,epoch,iteration):
    net_path = os.path.join("./net_state_dict","net_epoch_{}_iteration_{}.pth".format(epoch,iteration))
    if not os.path.exists("./net_state_dict"):
        os.makedirs("./net_state_dict")
    torch.save(net.state_dict(),net_path)
    print("第{}轮训练结果已经保存".format(epoch))

def save_checkpoint(model, optimizer, lr, epoch, model_folder):  # save model function

    model_out_path = model_folder + "net_epoch_{}.pth".format(epoch)

    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
        "lr": lr
    }
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
    torch.save(checkpoint, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def uint16to8(bands, lower_percent=0.001, higher_percent=99.999):
    out = np.zeros_like(bands, dtype=np.uint8)
    n = bands.shape[0]
    for i in range(n):
        a = 0  # np.min(band)
        b = 255  # np.max(band)
        c = np.percentile(bands[i, :, :], lower_percent)
        d = np.percentile(bands[i, :, :], higher_percent)

        t = a + (bands[i, :, :] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[i, :, :] = t
    return out


def getRGBImg(r,g,b,img_size=256):
    img=np.zeros((img_size,img_size,3),dtype=np.uint8)
    img[:,:,0]=r
    img[:,:,1]=g
    img[:,:,2]=b
    return img


def GetQuadrupletsImg(img_cld, img_fake, img_truth, img_csm, img_size=256, scale=2000):
    # print(img_cld.shape,img_fake.shape,img_truth.shape)
    output_img = np.zeros((img_size, 4 * img_size, 3), dtype=np.uint8)

    # 压缩维度 转NUMPY 转维度 乘以缩放比 再转int8
    # 转换之后维度分别为 256*256*15   256*256*15  256*256*15  256*256*1
    img_cld = uint16to8((t.squeeze(img_cld).cpu().numpy() * scale).astype("uint16")).transpose(1, 2, 0)
    img_fake = uint16to8((t.squeeze(img_fake).cpu().numpy() * scale).astype("uint16")).transpose(1, 2, 0)
    img_truth = uint16to8((t.squeeze(img_truth).cpu().numpy() * scale).astype("uint16")).transpose(1, 2, 0)

    img_csm = img_csm.cpu().numpy().transpose(1, 2, 0)
    # print(img_cld.shape,img_fake.shape,img_truth.ashape)
    # 取RGB
    img_cld_RGB = getRGBImg(img_cld[:, :, 3], img_cld[:, :, 2], img_cld[:, :, 1], img_size)
    img_fake_RGB = getRGBImg(img_fake[:, :, 3], img_fake[:, :, 2], img_fake[:, :, 1], img_size)
    img_truth_RGB = getRGBImg(img_truth[:, :, 3], img_truth[:, :, 2], img_truth[:, :, 1], img_size)
    # print(img_cld_RGB,img_fake_RGB,img_truth_RGB)
    # CSM转三通道
    img_csm_RGB = np.concatenate((img_csm, img_csm, img_csm), axis=-1) * 255

    # 合成！
    output_img[:, 0 * img_size:1 * img_size, :] = img_cld_RGB
    output_img[:, 1 * img_size:2 * img_size, :] = img_fake_RGB
    output_img[:, 2 * img_size:3 * img_size, :] = img_truth_RGB
    output_img[:, 3 * img_size:4 * img_size, :] = img_csm_RGB
    return output_img


#获得图片的原始数据
def get_raw_data(path):
    with rasterio.open(path, driver='GTiff') as src:
        image = src.read()

    # checkimage for nans
    image[np.isnan(image)] = np.nanmean(image)

    return image.astype('float32')

#获得RGB的数据或者SAR图片数据，用于后续处理
def get_rgb_preview(r, g, b, sar_composite=False):
    if not sar_composite:

        # stack and move to zero
        rgb = np.dstack((r, g, b))
        rgb = rgb - np.nanmin(rgb)

        # treat saturated images, scale values
        if np.nanmax(rgb) == 0 :
            rgb = 255 * np.ones_like(rgb)
        else:
            rgb = 255 * (rgb / np.nanmax(rgb))

        # replace nan values before final conversion
        rgb[np.isnan(rgb)] = np.nanmean(rgb)

        return rgb.astype(np.uint8)

    else:
        # generate SAR composite
        HH = r
        HV = g

        HH = np.clip(HH, -25.0, 0)
        HH = (HH + 25.1) * 255 / 25.1
        HV = np.clip(HV, -32.5, 0)
        HV = (HV + 32.6) * 255 / 32.6

        rgb = np.dstack((np.zeros_like(HH), HH, HV))

        return rgb.astype(np.uint8)

#CARL LOSS (critical)
def carl_error(y_true,csm, y_pred, batchsize):
    """Computes the Cloud-Adaptive Regularized Loss (CARL)"""
    
    clearmask = t.ones_like(csm) - csm
    predicted = y_pred
    input_cloudy = y_pred
    target = y_true
    if batchsize > 1:
        clearmask = torch.unsqueeze(clearmask, dim=1)
        csm = torch.unsqueeze(csm, dim=1)
    d = t.abs(predicted - input_cloudy)
    a = clearmask * d
    b = t.abs(predicted - target)
    c = t.mean(t.abs(predicted - target))


    #spatial_loss
    spatial_loss = Spatial_Loss(in_channels=13).cuda()
    sl = spatial_loss(predicted, target)
    cscmae = c + sl



    return cscmae






class Spatial_Loss(nn.Module):
    def __init__(self, in_channels):
        super(Spatial_Loss, self).__init__()
        self.res_scale = in_channels

        self.make_PAN = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

        self.L1_loss = nn.L1Loss().cuda()

    def forward(self, ref_HS, pred_HS):
        pan_pred = self.make_PAN(pred_HS)
        with torch.no_grad():
            pan_ref = self.make_PAN(ref_HS)
        spatial_loss = self.L1_loss(pan_pred, pan_ref)



        return spatial_loss