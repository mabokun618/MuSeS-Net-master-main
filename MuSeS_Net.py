import torch
import torch.nn.functional as F
from torch import nn

class ED(nn.Module):
    def __init__(self, in_channels):
        super(ED, self).__init__()
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fmap):
        batch_size, c, h, w = fmap.size()
        q, k, v = self.to_qkv(fmap).view(batch_size, -1, h * w).permute(0, 2, 1).chunk(3, dim=-1)
        cent_spec_vector = q[:, int((h * w - 1) / 2)]
        cent_spec_vector = torch.unsqueeze(cent_spec_vector, 1)
        csv_expand = cent_spec_vector.expand(batch_size, h * w, c)
        E_dist = torch.norm(csv_expand - k, dim=2, p=2)
        sim_E_dist = 1 / (1 + E_dist)
        atten_ED = self.softmax(sim_E_dist)
        atten_sim = torch.unsqueeze(atten_ED, 2)
        v_attened = torch.mul(atten_sim, v)
        out = v_attened.contiguous().view(batch_size, -1, h, w) + fmap
        return out


class CSS_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(CSS_Conv, self).__init__()
        self.point_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.depth_conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_channels,
        )
        self.leaky = nn.LeakyReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.BN = nn.BatchNorm2d(in_channels)
    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.leaky(out)
        out = self.depth_conv(out)
        out = self.relu(out)
        return out

# Semantic Euclidean Attention Module
class SEAM(nn.Module):
    def __init__(self, in_channels):
        super(SEAM, self).__init__()
        self.CSS_C_1 = CSS_Conv(in_channels, in_channels, 1)
        self.CSS_C_2 = CSS_Conv(in_channels, in_channels, 3)
        self.ED = ED(in_channels)
    def forward(self, x_fused):
        x1 = self.CSS_C_1(x_fused)
        x1 = self.CSS_C_2(x1)
        x = self.ED(x1)
        return x

# Hierarchical Pooling and Refinement Module
class HPRM(nn.Module): # pspnet中的金字塔池化模块
    def __init__(self,inchannel, down_dim):
        super(HPRM, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(inchannel, down_dim, 3, padding=1),
            nn.BatchNorm2d(down_dim),
            nn.PReLU()
        )
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(6, 6)),
            nn.Conv2d(down_dim, down_dim, kernel_size=1),
            nn.BatchNorm2d(down_dim),
            nn.PReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, kernel_size=1),
            nn.BatchNorm2d(down_dim),
            nn.PReLU()
        )
        self.bn_2d = nn.BatchNorm2d(48)
    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv1_up = F.upsample(conv1, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv1_up = self.bn_2d(conv1_up)
        return self.fuse(conv1_up)

class MuSeS(nn.Module):
    def __init__(self, in_channels, in_channels_fused, h, w, class_count):
        super(MuSeS, self).__init__()
        self.class_count = class_count
        self.in_channels = in_channels
        self.in_channels_fused = in_channels_fused
        self.height_p = h
        self.width_p = w
        self.win_spa_size = self.height_p * self.width_p
        self.SEAM = SEAM(in_channels_fused)
        self.fc = nn.Linear(48, self.class_count)
        self.HPRM = HPRM(self.in_channels_fused,down_dim=48)

    def forward(self, x1, x2):
        x_fused = torch.cat((x1, x2), dim=1)
        x = self.SEAM(x_fused)
        x = self.HPRM(x)
        x = x.mean(dim=(2, 3))
        out = self.fc(x)
        return out

