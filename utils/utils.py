# File: utils/utils.py
import torch
import os
import numpy as np
from pytorch_msssim import ssim as ssim_pt  # Using the same SSIM library


def findLastCheckpoint(save_dir):
    """Finds the latest epoch checkpoint in the save directory."""
    if not os.path.exists(save_dir):
        return None
    file_list = [f for f in os.listdir(save_dir) if f.startswith('net_epoch') and f.endswith('.pth')]
    if not file_list:
        return None

    epochs = []
    for f_name in file_list:
        try:
            # Extract epoch number, e.g., from "net_epoch10.pth"
            epoch_str = f_name.replace('net_epoch', '').replace('.pth', '')
            epochs.append(int(epoch_str))
        except ValueError:
            # Handle cases where filename might not be as expected, e.g. "net_best_psnr.pth"
            continue

    return max(epochs) if epochs else None


def batch_PSNR(img1, img2, data_range):
    """
    Calculates PSNR for a batch of images.
    img1, img2: PyTorch tensors, shape [B, C, H, W], range [0, data_range]
    data_range: Typically 1.0 for [0,1] or 255 for [0,255]
    """
    if not isinstance(img1, torch.Tensor):
        img1 = torch.tensor(img1)
    if not isinstance(img2, torch.Tensor):
        img2 = torch.tensor(img2)

    img1 = img1.to(torch.float32)
    img2 = img2.to(torch.float32)

    # Clamp values to the specified data_range, typically [0,1] if data_range is 1.0
    if data_range == 1.0:
        img1 = torch.clamp(img1, 0., 1.)
        img2 = torch.clamp(img2, 0., 1.)

    mse = torch.mean((img1 * (255.0 / data_range) - img2 * (255.0 / data_range)) ** 2,
                     dim=[1, 2, 3])  # Scale to 0-255 then MSE
    # Handle mse == 0 case (perfect match)
    psnr_val = torch.where(mse == 0, torch.tensor(float('inf'), device=mse.device),
                           10 * torch.log10((255.0 ** 2) / mse))

    # For inf PSNR, can cap at a high value like 100dB or handle as needed
    psnr_val = torch.where(torch.isinf(psnr_val), torch.tensor(100.0, device=mse.device), psnr_val)

    return torch.mean(psnr_val).item()


def batch_SSIM(img1_tensor, img2_tensor, data_range):
    """
    Calculates SSIM for a batch of images using pytorch_msssim.
    img1_tensor, img2_tensor: PyTorch tensors, shape [B, C, H, W], range [0, data_range]
    data_range: Typically 1.0 for [0,1] range.
    """
    if not isinstance(img1_tensor, torch.Tensor):
        img1_tensor = torch.tensor(img1_tensor)
    if not isinstance(img2_tensor, torch.Tensor):
        img2_tensor = torch.tensor(img2_tensor)

    img1_tensor = img1_tensor.to(torch.float32)
    img2_tensor = img2_tensor.to(torch.float32)

    if data_range == 1.0:  # Ensure inputs are [0,1] for ssim_pt if data_range=1.0
        img1_tensor = torch.clamp(img1_tensor, 0., 1.)
        img2_tensor = torch.clamp(img2_tensor, 0., 1.)

    # ssim_pt expects (N, C, H, W) and returns (N)
    # size_average=False returns per-image SSIM, then we average.
    ssim_val = ssim_pt(img1_tensor, img2_tensor, data_range=data_range, size_average=False)
    return torch.mean(ssim_val).item()
