import os
import sys
import torch as t
import argparse
import numpy as np

from dataloader import *
from Model_TCRNet import *
from generic_train_test import *
import img_operations as imgop
import time
import visualize
from lpips.lpips import LPIPS
import torch.nn.functional as F

##===================================================##
##********** Configure training settings ************##
##===================================================#
# PYTHON -m visdom.server

parser=argparse.ArgumentParser()
parser.add_argument('--stage', type=int, default=2)
parser.add_argument('--dim', type=int, default=32)
parser.add_argument('--dimsar', type=int, default=2)
parser.add_argument('--num_blocks', type=int, default=[2, 2, 2])
parser.add_argument('--batch_size', type=int, default=4, help='batch size used for training')

parser.add_argument('--data_list_filepath', type=str, default='E:\\cloud removal\\datafile_8000.csv')
# parser.add_argument('--data_list_filepath', type=str, default='F:\\cloud removal\\ROIs1158_spring-784\\data784.csv')
parser.add_argument('--is_use_cloudmask', type=bool, default=True)
parser.add_argument('--cloud_threshold', type=float, default=0.2) # only useful when is_use_cloudmask=True

parser.add_argument('--width', type=int, default=256)
parser.add_argument('--height', type=int, default=256)

parser.add_argument('--maxepoch', type=int, default=301)
parser.add_argument('--show_freq', type=int, default=1700) # 1700
parser.add_argument('--save_frequency', type=int, default=100)
parser.add_argument('--gpu_ids', type=str, default='0')
parser.add_argument('--log_freq', type=int, default=1)
parser.add_argument('--save_model_dir', type=str, default='D:\\pythonproject\\2-cloudremoval\\2023-SAR Spectral Transformer\\checkpoints', help='directory used to store trained networks')
parser.add_argument('--is_test', type=bool, default=False)
parser.add_argument('--input_data_folder', type=str, default='E:\\cloud removal\\ROIs1158_spring\\')
# parser.add_argument('--input_data_folder', type=str, default='F:\\cloud removal\\ROIs1158_spring-784\\')
parser.add_argument('--load_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of optimizer')
parser.add_argument('--beta1', type=int, default=0.9)
parser.add_argument('--output_dir', type=str, default='./output_imgs')
parser.add_argument('--lr_decay', type=int, default=0.00001, help='lr decay rate')
parser.add_argument('--optimizer', type=str, default='Adam', help = 'Adam')

parser.add_argument('--feature_sizes', type=int, default=256)
#parser.add_argument('--input_data_folder', type=str, default='E:\\pythonproject\\2-cloudremoval\\Dsen2-cr-pytorch-wmh\\data\\')
parser.add_argument('--alpha', type=int, default=0.1)
# parser.add_argument('--data_list_filepath', type=str, default='F:\\cloud removal\\ROIs1158_spring\\data.csv')
parser.add_argument('--lr_start_epoch_decay', type=int, default=5, help='epoch to start lr decay')

RESUME = False
epoch_checkpoint = 100
start_epoch = 1
opts = parser.parse_args()
# print_options(opts)

##===================================================##
##****************** choose gpu *********************##
##===================================================##
os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids

##===================================================##
##*************** Create dataloader *****************##
##===================================================##
imgop.seed_torch(42)

train_filelist, val_filelist, _ = get_train_val_test_filelists(opts.data_list_filepath)

train_data = AlignedDataset(opts, train_filelist)
train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=opts.batch_size, shuffle=True)

val_data = AlignedDataset(opts, val_filelist)
val_dataloader = torch.utils.data.DataLoader(dataset=val_data, batch_size=1, shuffle=True)
val_dataiter=iter(val_dataloader) # 创建迭代器对象，与next()一起用
##===================================================##
##****************** Create model *******************##
##===================================================##

#可视化操作


net = TCRNet(opts.dim, opts.dimsar, opts.stage, opts.num_blocks)

##===================================================##
##****************** load param *********************##
##===================================================##
# 优化器
optim_adam = t.optim.Adam(net.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999)) #, weight_decay=0.00001

scheduler = t.optim.lr_scheduler.StepLR(optim_adam, step_size=50, gamma=0.2)

if RESUME:
    path_checkpoint = opts.save_model_dir + "net_epoch_{}.pth".format(epoch_checkpoint)
    checkpoint = torch.load(path_checkpoint)
    net.load_state_dict(checkpoint['net'])
    net = net.cuda()
    optim_adam.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print('Network is Successfully Loaded from %s' % (path_checkpoint))

print('---------- Networks initialized -------------')
num_params = 0
for param in net.parameters():
    num_params += param.numel()
print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
print('-----------------------------------------------')


# 损失函数
CARL_Loss = imgop.carl_error

##===================================================##
##**************** Train the network ****************##
##===================================================##


for epoch in range(start_epoch, opts.maxepoch, 1):
    epoch_start_time = time.time()
    # 数据集的小循环
    for iteration, t_data in enumerate(train_dataloader, start=1):
        # _input = net.decode_input(data)
        cloudy_data, SAR_data, cloudfree_data, cloud_mask = net.set_input(t_data)

        net = net.cuda()

        pred_CloudFree_data = net(cloudy_data, SAR_data)
        optim_adam.zero_grad()
        loss = CARL_Loss(cloudfree_data, cloud_mask, pred_CloudFree_data, opts.batch_size)

        loss.backward()
        optim_adam.step()


        if iteration % opts.show_freq == 0:
            net.eval()
            psnr_13 = 0
            ssim_13 = 0
            with t.no_grad():
                print("epoch[{}]({}/{}):loss_fake:{:.8f}".format(
                    epoch, iteration, len(train_dataloader), loss.item()))

                for j, val_data_1 in enumerate(val_dataloader, 1):

                    val_cloudy_data, val_SAR_data, val_cloudfree_data, vali_cloud_mask = net.set_input(val_data_1)


                    val_pred_CloudFree_data = net(val_cloudy_data, val_SAR_data)
                    net.train()

                    psnr_1 = PSNR(val_pred_CloudFree_data, val_cloudfree_data)
                    psnr_13 = psnr_13 + psnr_1
                    ssim_1 = SSIM(val_pred_CloudFree_data, val_cloudfree_data).item()
                    ssim_13 = ssim_13 + ssim_1
                img_out = imgop.GetQuadrupletsImg(val_cloudy_data, val_pred_CloudFree_data, val_cloudfree_data,
                                                  vali_cloud_mask, img_size=opts.crop_size)

                psnr_13 = psnr_13/j
                ssim_13 = ssim_13/j
                print(iteration, '  psnr_13:', format(psnr_13, '.4f'), '  ssim_13:', format(ssim_13, '.4f'))
    print("learning rate:º%f" % (optim_adam.param_groups[0]['lr']))

    if epoch % opts.save_frequency == 0:
        imgop.save_checkpoint(net, optim_adam, opts.lr, epoch, opts.save_model_dir)


    print("第{}轮训练完毕，用时{}S".format(epoch, time.time() - epoch_start_time))


