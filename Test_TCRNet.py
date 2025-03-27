
import sys
import numpy as np

from dataloader import *
from Model_TCRNet import *
from generic_train_test import *
import img_operations as imgop
import time
from evaluation import *
import os
import torch
import argparse
import numpy as np
#from torchmetrics import SpectralAngleMapper
from metrics import PSNR, SSIM, SAM, ERGAS
import scipy.io as sio
from dataloader import AlignedDataset, get_train_val_test_filelists



##########################################################
def test(CR_net, opts):
    _, _, test_filelist = get_train_val_test_filelists(opts.data_list_filepath)

    data = AlignedDataset(opts, test_filelist)

    dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=opts.batch_size, shuffle=False)

    iters = 0

    psnrlist, ssimlist, ergaslist, samlist, cclist, radio_cloudlist = [], [], [], [], [], []

    for inputs in dataloader:
        cloudy_data = inputs['cloudy_data'].cuda()
        cloudfree_data1 = inputs['cloudfree_data'].cuda()
        cloudfree_data2 = cloudfree_data1.permute([0, 2, 3, 1]).squeeze(0).cpu().detach().numpy()
        cloudfree_data = np.array(cloudfree_data2, dtype=np.float32)
        SAR_data = inputs['SAR_data'].cuda()


        pred_cloudfree_data1 = CR_net(cloudy_data, SAR_data)
        pred_cloudfree_data2 = pred_cloudfree_data1.permute([0, 2, 3, 1]).squeeze(0).cpu().detach().numpy()
        pred_cloudfree_data = np.array(pred_cloudfree_data2, dtype=np.float32)

        PSNR_13 = PSNR(pred_cloudfree_data1, cloudfree_data1)
        psnrlist.append(np.round(PSNR_13, 4))
        SSIM_13 = SSIM(pred_cloudfree_data1, cloudfree_data1).item()
        ssimlist.append(np.round(SSIM_13, 4))
        msam = SAM(pred_cloudfree_data1, cloudfree_data1).item()
        samlist.append(np.round(msam, 4))
        ergas = ERGAS(pred_cloudfree_data1, cloudfree_data1).item()
        ergaslist.append(np.round(ergas, 4))


        cc = compute_cc(cloudfree_data, pred_cloudfree_data)
        cclist.append(np.round(cc, 4))
        iters = iters + 1
        print(iters, 'psnr:', format(PSNR_13, '.2f'), 'ssim:', format(SSIM_13, '.4f'), 'ergas:', format(ergas, '.4f'), 'sam:', format(msam, '.4f'), 'cc:', format(cc, '.4f'))

    psnr_mean = np.mean(psnrlist)
    ssim_mean = np.mean(ssimlist)
    ergas_mean = np.mean(ergaslist)
    msam_mean = np.mean(samlist)
    cc_mean = np.mean(cclist)
    print('psnr_mean:', format(psnr_mean, '.2f'), 'ssim_mean:', format(ssim_mean, '.4f'),'ergas_mean:', format(ergas_mean, '.4f'),'sam_mean:', format(msam_mean, '.4f'), 'cc_mean:', format(cc_mean,'.4f'))


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    parser = argparse.ArgumentParser()


    parser.add_argument('--stage', type=int, default=2)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--dimsar', type=int, default=2)
    parser.add_argument('--num_blocks', type=int, default=[2, 2, 2])
    parser.add_argument('--batch_size', type=int, default=1, help='batch size used for training')

    parser.add_argument('--feature_sizes', type=int, default=256)
    parser.add_argument('--alpha', type=int, default=0.1)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=256)

    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--input_data_folder', type=str, default='E:\\cloud removal\\ROIs1158_spring\\')
    parser.add_argument('--data_list_filepath', type=str, default='E:\\cloud removal\\datafile_8000.csv')

    parser.add_argument('--is_test', type=bool, default=True)
    parser.add_argument('--is_use_cloudmask', type=bool, default=True)
    parser.add_argument('--cloud_threshold', type=float, default=0.2)  # only useful when is_use_cloudmask=True

    opts = parser.parse_args()

    net = TCRNet(opts.dim, opts.dimsar, opts.stage, opts.num_blocks).cuda()
    checkpoint = torch.load('D:\\pythonproject\\2-cloudremoval\\2023-SAR Spectral Transformer\\result-SST-spatial_loss\\checkpointsnet_epoch_300.pth')
    net.load_state_dict(checkpoint['net'])

    net.eval()
    for _, param in net.named_parameters():
        param.requires_grad = False

    test(net, opts)


if __name__ == "__main__":
    main()
