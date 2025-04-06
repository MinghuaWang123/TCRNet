import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
import cv2
from einops import rearrange, repeat, reduce
import matplotlib.pyplot as plt


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class SobelConv2d(nn.Module):

    def __init__(self, in_channels=2, out_channels=4, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, requires_grad=True):
        assert kernel_size % 2 == 1, 'SobelConv2d\'s kernel_size must be odd.'
        assert out_channels % 2 == 0, 'SobelConv2d\'s out_channels must be a multiple of 4.'
        assert out_channels % groups == 0, 'SobelConv2d\'s out_channels must be a multiple of groups.'

        super(SobelConv2d, self).__init__()
        out_channels = 4
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # In non-trainable case, it turns into normal Sobel operator with fixed weight and no bias.
        self.bias = bias if requires_grad else False

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32), requires_grad=True)
        else:
            self.bias = None

        self.sobel_weight = nn.Parameter(torch.zeros(
            size=(out_channels, in_channels, kernel_size, kernel_size), dtype=torch.float32), requires_grad=False)


        # Initialize the Sobel kernal
        kernel_mid = kernel_size // 2
        for idx in range(out_channels):
            if idx % 2 == 0:
                self.sobel_weight[idx, :, 0, :] = -1
                self.sobel_weight[idx, :, 0, kernel_mid] = -2
                self.sobel_weight[idx, :, -1, :] = 1
                self.sobel_weight[idx, :, -1, kernel_mid] = 2
            elif idx % 2 == 1:
                self.sobel_weight[idx, :, :, 0] = -1
                self.sobel_weight[idx, :, kernel_mid, 0] = -2
                self.sobel_weight[idx, :, :, -1] = 1
                self.sobel_weight[idx, :, kernel_mid, -1] = 2


        # Define the trainable sobel factor
        if requires_grad:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=True)
        else:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=False)

    def forward(self, x):
        if torch.cuda.is_available():
            self.sobel_factor = self.sobel_factor.cuda()
            if isinstance(self.bias, nn.Parameter):
                self.bias = self.bias.cuda()

        sobel_weight = self.sobel_weight.cuda()

        if torch.cuda.is_available():
            sobel_weight = sobel_weight.cuda()

        out1 = F.conv2d(x, sobel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        G1 = torch.sqrt(torch.square(out1[:,0,:,:]) + torch.square(out1[:,1,:,:]))
        G2 = torch.sqrt(torch.square(out1[:, 2, :, :]) + torch.square(out1[:, 3, :, :]))
        out = torch.zeros(x.shape).cuda()
        out[:,0,:,:] = G1
        out[:, 1, :, :] = G2
        return out


class SARGuidedMechanism(nn.Module):
    def __init__(
            self, dim_SAR, n_feat, kSize, in_ch, sobel_ch):
        super(SARGuidedMechanism, self).__init__()

        self.conv1 = nn.Conv2d(dim_SAR*4, n_feat, kSize, padding=(kSize - 1) // 2, stride=1)
        self.conv_sobel = SobelConv2d(in_ch, sobel_ch, kernel_size=3, stride=1, padding=1, bias=True)

        self.depth_conv = nn.Conv2d(n_feat, n_feat, kernel_size=5, padding=2, bias=True, groups=n_feat)

        self.conv3 = nn.Conv2d(dim_SAR, dim_SAR, 3, padding=(3 - 1) // 2, stride=1)
        self.BN = nn.BatchNorm2d(dim_SAR)
        self.conv2 = nn.Conv2d(dim_SAR, n_feat, kernel_size=1, bias=True)
        self.conv32 = nn.Conv2d(dim_SAR, n_feat, 3, padding=(3 - 1) // 2, stride=1)




    def forward(self, SAR_img):
        # x: b,c,h,w
        [bs, nC, row, col] = SAR_img.shape
        a = self.conv3(SAR_img)
        SAR_cov3 = self.BN(a)


        SAR_Sobel = self.conv_sobel(SAR_cov3) + SAR_cov3
        SAR_edge1 = self.conv2(SAR_Sobel)
        SAR_edge2 = torch.sigmoid(self.conv32(SAR_Sobel))
        SAR_edge3 = self.depth_conv(self.conv2(SAR_Sobel))


        SAR_fea =SAR_edge1 +  SAR_edge2 + SAR_edge3

        return SAR_fea


class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_sar,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_m = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))  # sigma,将不可训练参数转为可训练，并绑定在Module里面
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.mm = SARGuidedMechanism(dim_sar, dim, 3, dim_sar, dim_sar*4)
        self.dim = dim

    def forward(self, x_in, SAR_img=None):
        """
        x_in: [b,h,w,c]
        mask: [1,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        mask_attn = self.mm(SAR_img).permute(0, 2, 3, 1)

        q, k, v, m_inp = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                             (q_inp, k_inp, v_inp, mask_attn.flatten(1, 2)))
        v = v * m_inp  # hadamard乘，逐元素相乘
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)  # l2 norm,
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q，矩阵乘法
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)  # nn.linear
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class MSAB(nn.Module):
    def __init__(
            self,
            dim,
            dimsar,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_sar=dimsar, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, SAR_img):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, SAR_img) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class TCRNet(nn.Module):
    def set_input(self, _input):
        inputs = _input
        self.cloudy_data = inputs['cloudy_data'].cuda()
        self.cloudfree_data = inputs['cloudfree_data'].cuda()
        self.SAR_data = inputs['SAR_data'].cuda()
        self.cloud_mask = inputs['cloud_mask'].cuda()
        return self.cloudy_data, self.SAR_data, self.cloudfree_data, self.cloud_mask

    def __init__(self, dim=13, dim_sar1=2, stage=3, num_blocks=[2, 2, 2]):
        super(TCRNet, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.embedding = nn.Conv2d(13, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        dim_sar = dim_sar1
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSAB(
                    dim=dim_stage, dimsar=dim_sar, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),  # feature downsample
                nn.Conv2d(dim_sar, dim_sar * 2, 4, 2, 1, bias=False)  # mask downsample
            ]))
            dim_stage *= 2
            dim_sar *= 2

        # Bottleneck
        self.bottleneck = MSAB(
            dim=dim_stage, dim_head=dim, dimsar=dim_sar, heads=dim_stage // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSAB(
                    dim=dim_stage // 2, dimsar=dim_sar // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2
            dim_sar //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, 13, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, SAR_img=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        if SAR_img == None:
            SAR_img = torch.zeros((4, 2, 128, 128)).cuda()

        # Embedding
        fea = self.lrelu(self.embedding(x))

        # Encoder
        fea_encoder = []
        SAR_imgs = []

        for (MSAB, FeaDownSample, MaskDownSample) in self.encoder_layers:
            fea = MSAB(fea, SAR_img)
            SAR_imgs.append(SAR_img)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            SAR_img = MaskDownSample(SAR_img)

        # Bottleneck
        fea = self.bottleneck(fea, SAR_img)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            SAR_img = SAR_imgs[self.stage - 1 - i]
            fea = LeWinBlcok(fea, SAR_img)

        # Mapping
        out = self.mapping(fea) + x

        return out




