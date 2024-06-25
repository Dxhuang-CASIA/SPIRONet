import torch
import torch.nn as nn
import torch.nn.functional as F


class channel_graph_interaction(nn.Module):
    def __init__(self, in_channels, device, img_size = 512, ratio = 4):
        super().__init__()

        self.device = device
        self.ratio = ratio

        self.pool = nn.MaxPool2d(ratio)
        self.relu = nn.ReLU(inplace = True)
        self.weight = nn.Parameter(torch.FloatTensor(int(img_size // ratio) ** 2, int(img_size // ratio) ** 2))
        self.pih_conv = nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1)
        self.graph_weight = nn.Conv2d(64, 64, kernel_size = 1, stride = 1, padding=0)
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=4)

    def forward(self, x):
        b, c, h, w = x.shape
        identity = x

        x = self.pool(x)

        x_pih = self.pih_conv(x).view(b, c, -1)
        x_pih = self.relu(x_pih)

        norm_x_phi = F.normalize(x_pih, p = 2, dim = 2)
        norm_x_phi_T = norm_x_phi.permute(0, 2, 1)
        A_tilde = torch.matmul(norm_x_phi, norm_x_phi_T)

        D_sqrt_inv = torch.zeros_like(A_tilde).to(self.device)
        diag_sum = torch.sum(A_tilde, 2)

        for i in range(diag_sum.shape[0]):
            diag_sqrt = 1.0 / torch.sqrt(diag_sum[i, :] + 1e-8)
            diag_sqrt[torch.isnan(diag_sqrt)] = 0
            diag_sqrt[torch.isinf(diag_sqrt)] = 0
            D_sqrt_inv[i, :, :] = torch.diag(diag_sqrt)

        I = torch.eye(D_sqrt_inv.shape[1]).to(self.device)
        I = I.repeat(D_sqrt_inv.shape[0], 1, 1)

        L_tilde = I - torch.matmul(torch.matmul(D_sqrt_inv, A_tilde), D_sqrt_inv)

        out = torch.matmul(torch.matmul(L_tilde, x.reshape(b, c, -1)), self.weight)
        out = out.reshape(b, c, int(h / self.ratio), int(w / self.ratio))
        out = self.up(out)
        out = self.relu(out + identity)
        return out