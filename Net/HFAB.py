import torch
import torch.nn as nn

"""
目标：用多层残差CBAM增强高频信息，利用注意力机制自适应关注重要特征。
"""

# ---------------- CBAM 部分 ---------------- #
"""
结合了两种注意力机制：
    1、通道注意力（Channel Attention）：通过全局平均池化和最大池化，学习哪个通道更重要。
    2、空间注意力（Spatial Attention）：通过“平均池化”和“最大池化”在通道维度，关注哪些空间位置更重要。

流程：
    输入特征图 x 经过通道注意力得到权重 x_channel_att，然后乘以 x。
    再经过空间注意力得到空间权重，乘以特征图。
    输出加权特征。
"""


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_planes, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


# ------------- 残差块带CBAM -------------- #

class ResidualCBAMBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.cbam = CBAM(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.cbam(out)
        return out + residual


# ------------ 高频分支多层残差CBAM块 --------------- #

class HighFrequencyAttentionBlock(nn.Module):
    """
    高频分支网络，每个方向包含多层残差CBAM块。
    """

    def __init__(self, num_blocks_each_dir):
        super().__init__()
        self.directions = 3  # HL/LH/HH
        self.channels_per_dir = 3
        mid_channels = 64  # 升维通道数

        # 每个方向用多层残差CBAM块
        self.dir_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.channels_per_dir, mid_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                *[ResidualCBAMBlock(mid_channels) for _ in range(num_blocks_each_dir)],
                nn.Conv2d(mid_channels, self.channels_per_dir, 3, padding=1)
            ) for _ in range(self.directions)
        ])
        # 如果in_channels==out_channels, 可Identity
        self.res_adapter = nn.Identity()

    def forward(self, x):
        # x: [B, 9, H, W]
        dir_features = []
        for i in range(self.directions):
            start = i * self.channels_per_dir
            end = (i + 1) * self.channels_per_dir
            dir_feature = x[:, start:end, :, :]  # [B,3,H,W]
            processed = self.dir_convs[i](dir_feature)
            dir_features.append(processed)
        fused = torch.cat(dir_features, dim=1)
        output = fused + self.res_adapter(x)  # 残差
        return output  # [B,9,H,W]

# # --------------- 测试代码 ------------------- #
#
# if __name__ == "__main__":
#     dummy_input = torch.randn(2, 9, 128, 128)
#     model = HighFrequencyAttentionBlock(num_blocks_each_dir=3)
#     out = model(dummy_input)
#     print(f"输出形状: {out.shape}")  # [2,9,128,128]
#     print(f"参数量(M): {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
