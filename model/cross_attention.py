import torch
import torch.nn as nn
import torch.nn.functional as F


class PSPModule(nn.Module):
    def __init__(self, sizes=(1, 3, 6, 8), dimension = 2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center


class Attention_Module(nn.Module):
    def __init__(self, in_channels, ratio = 8):
        super(Attention_Module, self).__init__()

        self.key_embed = in_channels // ratio

        self.q_x_s = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size = 1),
            nn.BatchNorm2d(in_channels // ratio),
            nn.ReLU(inplace = True)
        )
        self.q_x_f = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size = 1),
            nn.BatchNorm2d(in_channels // ratio),
            nn.ReLU(inplace = True)
        )

        self.k_x_s = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size = 1),
            nn.BatchNorm2d(in_channels // ratio),
            nn.ReLU(inplace = True)
        )
        self.k_x_f = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size = 1),
            nn.BatchNorm2d(in_channels // ratio),
            nn.ReLU(inplace = True)
        )

        self.v = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels // ratio, kernel_size = 1),
            nn.BatchNorm2d(in_channels // ratio),
            nn.ReLU(inplace = True)
        )

        self.Conv_convert = nn.Conv2d(in_channels // ratio, in_channels, kernel_size = 1, stride = 1)
        self.psp = PSPModule(sizes = (1, 3, 6, 8))
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x_s, x_f):
        B, C, H, W = x_s.shape

        query_x_s = self.q_x_s(x_s).view(B, self.key_embed, -1).permute(0, 2, 1)
        query_x_f = self.q_x_f(x_f).view(B, self.key_embed, -1).permute(0, 2, 1)

        key_x_s = self.psp(self.k_x_s(x_s))
        key_x_f = self.psp(self.k_x_f(x_f))

        value = self.psp(self.v(torch.cat([x_s, x_f], dim = 1))).permute(0, 2, 1)

        score1 = torch.matmul(query_x_s, key_x_f)
        score1 = (self.key_embed ** -0.5) * score1

        score2 = torch.matmul(query_x_f, key_x_s)
        score2 = (self.key_embed ** -0.5) * score2

        score = F.softmax(score1 + score2, dim = -1) # /2?

        context = torch.matmul(score, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(B, self.key_embed, H, W)

        context = self.Conv_convert(context)

        context = context + x_f + x_s

        return self.relu(context)

if __name__ == '__main__':
    x = torch.randn(1, 64, 512, 512)
    model = Attention_Module(in_channels = 64)
    out = model(x, x)