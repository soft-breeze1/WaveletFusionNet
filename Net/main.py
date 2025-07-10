import torch
import torch.nn as nn
from Net.LFTU import LowFrequencyTransformerUNet
from Net.HFAB import HighFrequencyAttentionBlock
from utils.wavelet import DWT, IWT

class WaveletFusionNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 embed_dim=256,    # 对应LLTransformerBranch的embed_dim
                 depth=12,
                 num_heads=8,
                 mlp_ratio=4.0):
        super(WaveletFusionNet, self).__init__()
        self.dwt = DWT()
        self.iwt = IWT()
        # 低频分支
        self.low_freq_branch = LowFrequencyTransformerUNet(
            in_channels=in_channels,
            base_channels=base_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )
        self.high_freq_branch = HighFrequencyAttentionBlock(num_blocks_each_dir=3)

    def forward(self, x):
        dwt_out = self.dwt(x)  # [B, 12, H/2, W/2]
        ll = dwt_out[:, 0:3, :, :]   # 低频 LL
        hl = dwt_out[:, 3:6, :, :]   # 高频 HL
        lh = dwt_out[:, 6:9, :, :]   # 高频 LH
        hh = dwt_out[:, 9:12, :, :]  # 高频 HH
        high_freq = torch.cat([hl, lh, hh], dim=1)
        enhanced_ll = self.low_freq_branch(ll)
        enhanced_hf = self.high_freq_branch(high_freq)
        fused_wavelet = torch.cat([enhanced_ll, enhanced_hf], dim=1)
        enhanced_img = self.iwt(fused_wavelet)

        return enhanced_img
