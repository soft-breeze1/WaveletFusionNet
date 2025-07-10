import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from torchvision.models import vgg16
from utils.wavelet import DWT

# -------- 本地载入VGG特征 -------- #
def load_vgg_features(model_path):
    vgg = vgg16(weights=None)  # 不自动下载
    state_dict = torch.load(model_path)
    vgg.load_state_dict(state_dict)
    features = vgg.features[:16].eval()
    for p in features.parameters():
        p.requires_grad = False
    return features

# -------- VGG感知损失（本地权重） -------- #
class VGGPerceptualLoss(nn.Module):
    def __init__(self, vgg_model_path=r"D:\EI\Net-2\vgg16-397923af.pth"):
        super().__init__()
        self.vgg = load_vgg_features(vgg_model_path)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        feat_input = self.vgg(input)
        feat_target = self.vgg(target)
        return F.l1_loss(feat_input, feat_target)

# -------- 颜色一致性损失（RGB通道均值） -------- #
def color_constancy_loss(output, target):
    mean_output = output.mean(dim=[2, 3], keepdim=True)
    mean_target = target.mean(dim=[2, 3], keepdim=True)
    return F.l1_loss(mean_output, mean_target)

# -------- Edge Loss -------- #
def edge_loss(pred, target):
    sobel_kernel_x = torch.tensor([[1, 0, -1],
                                   [2, 0, -2],
                                   [1, 0, -1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3) / 8.0
    sobel_kernel_y = torch.tensor([[1, 2, 1],
                                   [0, 0, 0],
                                   [-1, -2, -1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3) / 8.0

    grad_pred_x = F.conv2d(pred, sobel_kernel_x.expand(3, 1, 3, 3), padding=1, groups=3)
    grad_pred_y = F.conv2d(pred, sobel_kernel_y.expand(3, 1, 3, 3), padding=1, groups=3)
    grad_tgt_x = F.conv2d(target, sobel_kernel_x.expand(3, 1, 3, 3), padding=1, groups=3)
    grad_tgt_y = F.conv2d(target, sobel_kernel_y.expand(3, 1, 3, 3), padding=1, groups=3)

    grad_pred = torch.sqrt(grad_pred_x ** 2 + grad_pred_y ** 2 + 1e-6)
    grad_tgt = torch.sqrt(grad_tgt_x ** 2 + grad_tgt_y ** 2 + 1e-6)
    return F.l1_loss(grad_pred, grad_tgt)

# -------- WaveletFusionLoss with MS-SSIM & Edge & Local VGG -------- #
class WaveletFusionLoss(nn.Module):
    def __init__(self,
                 lambda_img=1.0,
                 lambda_ll=1.2,
                 lambda_hf=0.3,
                 lambda_vgg=0.05,
                 lambda_ssim=2.0,
                 lambda_color=0.01,
                 lambda_edge=0.1,
                 vgg_model_path=r"D:\EI\WaveletFusionNet-main\vgg16-397923af.pth"):
        super().__init__()
        self.dwt = DWT()
        self.l1 = nn.L1Loss()
        self.vgg = VGGPerceptualLoss(vgg_model_path)
        self.lambda_img = lambda_img
        self.lambda_ll = lambda_ll
        self.lambda_hf = lambda_hf
        self.lambda_vgg = lambda_vgg
        self.lambda_ssim = lambda_ssim
        self.lambda_color = lambda_color
        self.lambda_edge = lambda_edge

    def forward(self, output_img, target_img):
        # 小波变换
        output_dwt = self.dwt(output_img)
        target_dwt = self.dwt(target_img)
        out_ll, out_hf = output_dwt[:, 0:3], output_dwt[:, 3:12]
        tgt_ll, tgt_hf = target_dwt[:, 0:3], target_dwt[:, 3:12]

        # 各种损失项
        loss_img = self.l1(output_img, target_img)
        loss_ll = self.l1(out_ll, tgt_ll)
        loss_hf = self.l1(out_hf, tgt_hf)
        loss_vgg = self.vgg(output_img, target_img)
        loss_ssim = 1 - ms_ssim(output_img, target_img, data_range=1.0, size_average=True)
        loss_color = color_constancy_loss(output_img, target_img)
        loss_edge = edge_loss(output_img, target_img)

        # 总损失加权
        loss = (self.lambda_img * loss_img +
                self.lambda_ll * loss_ll +
                self.lambda_hf * loss_hf +
                self.lambda_vgg * loss_vgg +
                self.lambda_ssim * loss_ssim +
                self.lambda_color * loss_color +
                self.lambda_edge * loss_edge)

        return loss, {
            'loss_img': loss_img.item(),
            'loss_ll': loss_ll.item(),
            'loss_hf': loss_hf.item(),
            'loss_vgg': loss_vgg.item(),
            'loss_ssim': loss_ssim.item(),
            'loss_color': loss_color.item(),
            'loss_edge': loss_edge.item(),
        }
