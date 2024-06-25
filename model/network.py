import torch
import torch.nn as nn
import torch.fft as fft
from torch.nn import functional as F
from .cross_attention import Attention_Module
from .graph_module import channel_graph_interaction


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = self.conv3(x)
        out = out + residual
        return self.relu(out)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim = 1)
        x = self.conv(x)
        return x


class FrequencyFilter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.amp_mask = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1, inplace = True),
            nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1)
        )
        self.pha_mask = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1, inplace = True),
            nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1)
        )

        self.channel_adjust = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        b, c, h, w = x.shape
        msF = fft.rfft2(x + 1e-8, norm = 'backward')

        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)

        amp_fuse = self.amp_mask(msF_amp) + msF_amp
        pha_fuse = self.pha_mask(msF_pha) + msF_pha

        real = amp_fuse * torch.cos(pha_fuse) + 1e-8
        imag = amp_fuse * torch.sin(pha_fuse) + 1e-8

        out = torch.complex(real, imag) + 1e-8
        out = torch.abs(torch.fft.irfft2(out, s = (h, w), norm = 'backward'))
        out = out + x
        out = self.channel_adjust(out)
        out = torch.nan_to_num(out, nan = 1e-5, posinf = 1e-5, neginf = 1e-5)

        return out


class SpatialEnc(nn.Module):
    def __init__(self, in_channels):
        super(SpatialEnc, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.enc_blk1 = DoubleConv(in_channels, out_channels = 64)
        self.enc_blk2 = DoubleConv(in_channels = 64, out_channels = 128)
        self.enc_blk3 = DoubleConv(in_channels = 128, out_channels = 256)
        self.enc_blk4 = DoubleConv(in_channels = 256, out_channels = 512)

    def forward(self, x):
        x0 = self.enc_blk1(x) # [64, H, W]

        x1 = self.maxpool(x0) # [64, H/2, W/2]
        x1 = self.enc_blk2(x1) # [128, H/2, W/2]

        x2 = self.maxpool(x1) # [128, H/4, W/4]
        x2 = self.enc_blk3(x2) # [256, H/4, W/4]

        x3 = self.maxpool(x2) # [256, H/8, W/8]
        x3 = self.enc_blk4(x3) # [512, H/8, W/8]

        x4 = self.maxpool(x3) # [512, H/16, W/16]

        return x0, x1, x2, x3, x4


class FrequencyEnc(nn.Module):
    def __init__(self, in_channels):
        super(FrequencyEnc, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.enc_blk1 = FrequencyFilter(in_channels, out_channels = 64)
        self.enc_blk2 = FrequencyFilter(in_channels = 64, out_channels = 128)
        self.enc_blk3 = FrequencyFilter(in_channels = 128, out_channels = 256)
        self.enc_blk4 = FrequencyFilter(in_channels = 256, out_channels = 512)

    def forward(self, x):
        x0 = self.enc_blk1(x) # [64, H, W]

        x1 = self.maxpool(x0) # [64, H/2, W/2]
        x1 = self.enc_blk2(x1) # [128, H/2, W/2]

        x2 = self.maxpool(x1) # [128, H/4, W/4]
        x2 = self.enc_blk3(x2) # [256, H/4, W/4]

        x3 = self.maxpool(x2) # [256, H/8, W/8]
        x3 = self.enc_blk4(x3) # [512, H/8, W/8]

        x4 = self.maxpool(x3) # [512, H/16, W/16]

        return x0, x1, x2, x3, x4


class Segmodel(nn.Module):
    def __init__(self, in_channels, num_classes, img_size, device):
        super(Segmodel, self).__init__()

        self.spatial_enc = SpatialEnc(in_channels)
        self.frequency_enc = FrequencyEnc(in_channels)

        self.cross_attn0 = Attention_Module(in_channels = 64)
        self.cross_attn1 = Attention_Module(in_channels = 128)
        self.cross_attn2 = Attention_Module(in_channels = 256)
        self.cross_attn3 = Attention_Module(in_channels = 512)
        self.cross_attn4 = Attention_Module(in_channels = 512)

        self.dec4 = Up(in_channels = 1024, out_channels = 256)
        self.dec3 = Up(in_channels = 512, out_channels = 128)
        self.dec2 = Up(in_channels = 256, out_channels = 64)
        self.dec1 = Up(in_channels = 128, out_channels = 64)
        self.dec0 = channel_graph_interaction(in_channels = 64, device = device, img_size = img_size) # 300 !!!!!!!!!!!

        self.head = nn.Conv2d(in_channels = 64, out_channels = num_classes, kernel_size = 1)

    def forward(self, x):
        x_s_0, x_s_1, x_s_2, x_s_3, x_s_4 = self.spatial_enc(x)
        x_f_0, x_f_1, x_f_2, x_f_3, x_f_4 = self.frequency_enc(x)

        x_4 = self.cross_attn4(x_s_4, x_f_4)
        x_3 = self.cross_attn3(x_s_3, x_f_3)
        x_2 = self.cross_attn2(x_s_2, x_f_2)
        x_1 = self.cross_attn1(x_s_1, x_f_1)
        x_0 = self.cross_attn0(x_s_0, x_f_0)

        x3 = self.dec4(x_4, x_3)
        x2 = self.dec3(x3, x_2)
        x1 = self.dec2(x2, x_1)
        x0 = self.dec1(x1, x_0)
        x0 = self.dec0(x0)

        out = self.head(x0)

        return out


if __name__ == '__main__':
    x = torch.randn(1, 1, 512, 512)
    model = Segmodel(in_channels = 1, num_classes = 1, device = 'cuda: 0')
    print(model(x).shape)