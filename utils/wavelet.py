import torch
import torch.nn as nn


# import torchvision.transforms as T
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np


def Normalize(x):
    ymax = 255
    ymin = 0
    xmax = x.max()
    xmin = x.min()
    return (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin


def dwt_init(x):
    """
    以通道维度拼接小波变换结果:
    输入: [B, 3, H, W]
    输出: [B, 12, H/2, W/2] -> [LL(3), HL(3), LH(3), HH(3)]
    """
    # 隔行采样并缩放
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2

    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    # 计算 4 个子带，每个子带都是 [B, 3, H/2, W/2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    # 在通道维度拼接：得到 [B, 12, H/2, W/2]
    return torch.cat((x_LL, x_HL, x_LH, x_HH), dim=1)


def iwt_init(x):
    """
    通道拼接版逆小波变换:
    输入: [B, 12, H/2, W/2] -> [LL(3), HL(3), LH(3), HH(3)]
    输出: [B, 3, H, W]
    """
    B, C, H, W = x.shape
    assert C == 12, "输入通道数必须为 12（3+3+3+3）"

    # 拆分子带
    x1 = x[:, 0:3, :, :] / 2  # LL
    x2 = x[:, 3:6, :, :] / 2  # HL
    x3 = x[:, 6:9, :, :] / 2  # LH
    x4 = x[:, 9:12, :, :] / 2  # HH

    # 初始化重建图像
    out = torch.zeros([B, 3, H * 2, W * 2], device=x.device)

    # 重建
    out[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    out[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    out[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    out[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return out


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


# def visualize_subbands(image_tensor):
#     dwt = DWT()
#     iwt = IWT()
#
#     # 原图归一化到 [0, 1] 显示
#     orig = image_tensor.clone().detach().cpu()
#     orig_np = orig.squeeze(0).permute(1, 2, 0).numpy()
#
#     # 小波变换
#     x_dwt = dwt(image_tensor)  # [B, 12, H/2, W/2]
#     LL = x_dwt[:, 0:3]
#     HL = x_dwt[:, 3:6]
#     LH = x_dwt[:, 6:9]
#     HH = x_dwt[:, 9:12]
#
#     # 逆变换
#     x_recon = iwt(x_dwt)
#     recon_np = x_recon.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
#
#     # 可视化低频 & 高频子带
#     def tensor2img(t):  # 3xHxW -> HxWx3 numpy
#         return Normalize(t.squeeze(0)).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
#
#     fig, axs = plt.subplots(2, 3, figsize=(15, 10))
#
#     axs[0, 0].imshow(orig_np.astype(np.uint8))
#     axs[0, 0].set_title("Original Image")
#     axs[0, 0].axis('off')
#
#     axs[0, 1].imshow(tensor2img(LL))
#     axs[0, 1].set_title("Low-Frequency (LL)")
#     axs[0, 1].axis('off')
#
#     axs[0, 2].imshow(recon_np.astype(np.uint8))
#     axs[0, 2].set_title("Reconstructed Image")
#     axs[0, 2].axis('off')
#
#     axs[1, 0].imshow(tensor2img(HL))
#     axs[1, 0].set_title("High-Frequency (HL)")
#     axs[1, 0].axis('off')
#
#     axs[1, 1].imshow(tensor2img(LH))
#     axs[1, 1].set_title("High-Frequency (LH)")
#     axs[1, 1].axis('off')
#
#     axs[1, 2].imshow(tensor2img(HH))
#     axs[1, 2].set_title("High-Frequency (HH)")
#     axs[1, 2].axis('off')
#
#     plt.tight_layout()
#     plt.show()
#
# # 示例：读取一张图片并测试
# def test_wavelet_on_image(img_path):
#     transform = T.Compose([
#         T.Resize((256, 256)),
#         T.ToTensor(),
#         lambda x: x * 255  # 保持与 DWT 输入一致
#     ])
#
#     img = Image.open(img_path).convert("RGB")
#     img_tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
#     visualize_subbands(img_tensor)
#
# # 用法示例
# test_wavelet_on_image(r"D:\EI\Net-2\data\UIEB\test\input\796_img_.png")