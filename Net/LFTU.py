from utils.unet import inconv, down, up, outconv
import torch
import torch.nn as nn
import kornia.color as kc

class MultiColorSpaceFusion(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # 初始化函数，接受输入通道数（默认为3）
    def forward(self, x):
        # 前向传播，将输入图像转换为多色彩空间并拼接
        x_rgb = x  # 原始RGB
        x_hsv = kc.rgb_to_hsv(x)  # 转换为HSV
        x_lab = kc.rgb_to_lab(x)  # 转换为Lab
        out = torch.cat([x_rgb, x_hsv, x_lab], dim=1)  # 在通道维拼接，共9通道
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        # Transformer的基本块，包含归一化、注意力和MLP
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)  # 多头注意力，batch_first表示输入(batch, seq, feature)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),  # MLP第一层扩大
            nn.GELU(),  # 激活函数
            nn.Linear(int(dim*mlp_ratio), dim)  # MLP第二层恢复
        )
    def forward(self, x):
        # 前向传播
        # 1. 归一化后执行注意力
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # 2. 归一化后通过MLP，再加残差
        x = x + self.mlp(self.norm2(x))
        return x

def flatten_hw(x):  # [B,C,H,W]→[B,H*W,C]
    # 将空间维H*W展平为序列长度，流程上是Transformer接受的输入格式
    x = x.flatten(2).transpose(1,2)
    return x

def unflatten_hw(x, H, W): # [B,H*W,C]→[B,C,H,W]
    B, L, C = x.shape  # 反向还原
    return x.transpose(1,2).reshape(B, C, H, W)

class LowFrequencyTransformerUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, embed_dim=256, depth=12, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        # 初始化整个网络
        self.color_fusion = MultiColorSpaceFusion(in_channels)  # 多色彩空间融合模块
        # 编码器部分
        self.inc = inconv(9, base_channels)  # 初始卷积，输入9通道
        self.down1 = down(base_channels, base_channels * 2)  # 下采样1
        self.down2 = down(base_channels * 2, base_channels * 4)  # 下采样2
        self.down3 = down(base_channels * 4, embed_dim)  # 下采样至Transformer输入维度
        # Transformer部分，多个块堆叠
        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        # 解码部分（反卷积上采样）
        self.up1 = up(embed_dim + base_channels * 4, base_channels * 4)  # 上采样1，注意：输入拼接后通道数
        self.up2 = up(base_channels * 4 + base_channels * 2, base_channels * 2)  # 上采样2
        self.up3 = up(base_channels * 2 + base_channels, base_channels)  # 上采样3
        self.outc = outconv(base_channels, 3)  # 输出层，恢复到3通道RGB
    def forward(self, ll):
        # 前向传播流程
        fused = self.color_fusion(ll)             # 1. 多色空间融合，输出[B,9,H,W]
        e0 = self.inc(fused)                       # 2. 初卷积特征
        e1 = self.down1(e0)                        # 3. 下采样1
        e2 = self.down2(e1)                        # 4. 下采样2
        e3 = self.down3(e2)                        # 5. 下采样3（bottleneck，变成Transformer的输入）
        B, tdim, H_, W_ = e3.shape
        tokens = flatten_hw(e3)                   # 6. 展平空间维度为序列（[B, H'*W', embed_dim]）
        tokens = self.transformer_blocks(tokens)  # 7. 经过Transformer编码
        e3 = unflatten_hw(tokens, H_, W_)         # 8. 转换回空间特征（保持空间结构）
        d1 = self.up1(e3, e2)                      # 9. 反卷积+拼接上采样
        d2 = self.up2(d1, e1)                      # 10. 上采样
        d3 = self.up3(d2, e0)                      # 11. 上采样
        out = self.outc(d3)                        # 12. 生成最终输出图像
        return out

# usage示例（注释）
# if __name__ == '__main__':
#     B,C,H,W = 2,3,290,890
#     dummy_ll = torch.rand(B,C,H,W)
#     model = LowFrequencyTransformerUNet(in_channels=3)
#     with torch.no_grad():
#         out_ll = model(dummy_ll)
#     print(out_ll.shape)
