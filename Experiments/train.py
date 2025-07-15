import sys
sys.path.append('/home/label/bzhang/pyproject/WaveletFusionNet-main/')

import random
import numpy as np
import torch
import os
import time
from tqdm import tqdm
import argparse
from torch import nn
from Net.main import WaveletFusionNet
from Loss.JointLoss import WaveletFusionLoss
import torch.optim as optim
from utils.utils import batch_PSNR, batch_SSIM
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from utils.data_RGB import get_training_data, get_validation_data
from utils.UIQM_UCIQE import calculate_uiqm, calculate_uciqe
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args():
    parser = argparse.ArgumentParser(description="WaveletFusionNet_train")
    parser.add_argument("--batchSize", type=int, default=16)
    parser.add_argument("--patchSize", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_weights", type=str, default=r"D:\EI\WaveletFusionNet-main\models\model_UIEB")
    parser.add_argument("--train_data", type=str, default=r"D:\EI\WaveletFusionNet-main\data\UIEB\train")
    parser.add_argument("--val_data", type=str, default=r"D:\EI\WaveletFusionNet-main\data\UIEB\Validation")
    parser.add_argument("--use_GPU", type=bool, default=True)
    parser.add_argument("--decay", type=int, default=25)
    return parser.parse_args()

def print_and_log(logf, s, print_on_screen=True):
    if print_on_screen:
        print(s)
    with open(logf, 'a+', encoding="utf-8") as f:
        f.write(s + '\n')

def get_UIQM_UCIQE(preds, crop_border=0):
    preds = preds.detach().cpu().numpy()
    B = preds.shape[0]
    uiqm_val, uciqe_val = 0., 0.
    for i in range(B):
        img = preds[i].transpose(1, 2, 0)
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        uiqm_val += calculate_uiqm(img, None, crop_border)
        uciqe_val += calculate_uciqe(img, None, crop_border)
    return uiqm_val / B, uciqe_val / B

if __name__ == '__main__':
    opt = get_args()

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    train_dataset = get_training_data(opt.train_data, {'patch_size': opt.patchSize})
    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=1, drop_last=False, pin_memory=True)

    val_dataset = get_validation_data(opt.val_data, {'patch_size': opt.patchSize})
    val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)

    model = WaveletFusionNet()
    criterion = WaveletFusionLoss()

    if opt.use_GPU:
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    milestones = [i for i in range(1, opt.epochs + 1) if i % opt.decay == 0]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    save_dir = opt.save_weights
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_file = os.path.join(save_dir, 'train_log.txt')

    # 加载断点
    checkpoint_path = os.path.join(save_dir, 'net_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        initial_epoch = ckpt.get('epoch', 0)
        print(f"Resumed from checkpoint at epoch {initial_epoch}")
    else:
        initial_epoch = 0

    best_psnr = 0
    best_ssim = 0
    best_uiqm = 0
    best_uciqe = 0
    fine_tune_epochs = 400
    fine_tune_start = opt.epochs - fine_tune_epochs
    fine_tune_lr = 1e-5
    switched = False

    if not os.path.exists(log_file) or initial_epoch == 0:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("Epoch,Loss_total,Loss_img,Loss_ll,Loss_hf,Loss_vgg,Loss_ssim,Loss_color,Loss_edge,PSNR,Best_PSNR,SSIM,Best_SSIM,UIQM,Best_UIQM,UCIQE,Best_UCIQE\n")

    # ====================== 训练和验证循环 ==========================
    for epoch in range(initial_epoch, opt.epochs):
        if not switched and epoch >= fine_tune_start:
            print_and_log(log_file, f"\n[Fine-tuning] Set LR to {fine_tune_lr} from epoch {epoch+1}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = fine_tune_lr
            switched = True

        model.train()
        epoch_loss = 0
        loss_items = {k: 0 for k in ['total', 'img', 'll', 'hf', 'vgg', 'ssim', 'color', 'edge']}
        train_bar = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}', leave=True)
        for i, (target, input_, _) in enumerate(train_bar):
            optimizer.zero_grad()
            if opt.use_GPU:
                input_ = input_.cuda()
                target = target.cuda()
            output = model(input_)
            loss, loss_dict = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            for key in loss_items:
                if f'loss_{key}' in loss_dict:
                    loss_items[key] += loss_dict[f'loss_{key}']
            train_bar.set_postfix({
                'loss': f'{epoch_loss / (i + 1):.4f}',
                'img': f'{loss_items["img"] / (i + 1):.4f}',
                'll': f'{loss_items["ll"] / (i + 1):.4f}',
                'color': f'{loss_items["color"] / (i + 1):.4f}',
                'edge': f'{loss_items["edge"] / (i + 1):.4f}',
            })

        scheduler.step()

        # ----------------- 验证阶段 ------------------
        model.eval()
        psnr_val_rgb = 0
        ssim_val_rgb = 0
        uiqm_val = 0
        uciqe_val = 0
        num_val_batches = len(val_loader)
        val_bar = tqdm(val_loader, desc=f'Validating Epoch {epoch + 1}', leave=False)
        for ii, (target, input_, _) in enumerate(val_bar):
            if opt.use_GPU:
                input_ = input_.cuda()
                target = target.cuda()
            with torch.no_grad():
                restored = torch.clamp(model(input_), 0., 1.)
            psnr_val_rgb += batch_PSNR(restored, target, 1.)
            ssim_val_rgb += batch_SSIM(restored, target, 1)
            uiqm_batch, uciqe_batch = get_UIQM_UCIQE(restored)
            uiqm_val += uiqm_batch
            uciqe_val += uciqe_batch
            val_bar.set_postfix({
                'PSNR': f'{psnr_val_rgb / (ii + 1):.2f}',
                'SSIM': f'{ssim_val_rgb / (ii + 1):.4f}',
                'UIQM': f'{uiqm_val / (ii + 1):.3f}',
                'UCIQE': f'{uciqe_val / (ii + 1):.3f}'
            })

        epoch_psnr = psnr_val_rgb / num_val_batches
        epoch_ssim = ssim_val_rgb / num_val_batches
        epoch_uiqm = uiqm_val / num_val_batches
        epoch_uciqe = uciqe_val / num_val_batches

        is_best = epoch_psnr > best_psnr
        if is_best:
            best_psnr = epoch_psnr
            # 保存当前最优 PSNR 的模型
            best_model_path = os.path.join(save_dir, 'net_best_psnr.pth')
            torch.save(model.state_dict(), best_model_path)
            print_and_log(log_file, f">>>> Saved best PSNR model at epoch {epoch+1}, PSNR={epoch_psnr:.2f}")

        best_ssim = max(best_ssim, epoch_ssim)
        best_uiqm = max(best_uiqm, epoch_uiqm)
        best_uciqe = max(best_uciqe, epoch_uciqe)

        avg_loss = epoch_loss / len(train_loader)
        summary_str = (
            f"Epoch {epoch + 1}: "
            f"Loss:[Total:{avg_loss:.4f}, Img:{loss_items['img']/len(train_loader):.4f}, LL:{loss_items['ll']/len(train_loader):.4f}, "
            f"HF:{loss_items['hf']/len(train_loader):.4f}, VGG:{loss_items['vgg']/len(train_loader):.4f}, "
            f"SSIM:{loss_items['ssim']/len(train_loader):.4f}, Color:{loss_items['color']/len(train_loader):.4f}, "
            f"Edge:{loss_items['edge']/len(train_loader):.4f}] "
            f"PSNR:{epoch_psnr:.2f} (Best:{best_psnr:.2f}), SSIM:{epoch_ssim:.4f} (Best:{best_ssim:.4f}), "
            f"UIQM:{epoch_uiqm:.3f} (Best:{best_uiqm:.3f}), UCIQE:{epoch_uciqe:.3f} (Best:{best_uciqe:.3f})"
        )
        print_and_log(log_file, summary_str)

        log_line = (
            f"{epoch+1},{avg_loss:.4f},{loss_items['img']/len(train_loader):.4f},{loss_items['ll']/len(train_loader):.4f},"
            f"{loss_items['hf']/len(train_loader):.4f},{loss_items['vgg']/len(train_loader):.4f},{loss_items['ssim']/len(train_loader):.4f},"
            f"{loss_items['color']/len(train_loader):.4f},{loss_items['edge']/len(train_loader):.4f},"
            f"{epoch_psnr:.2f},{best_psnr:.2f},{epoch_ssim:.4f},{best_ssim:.4f},{epoch_uiqm:.3f},{best_uiqm:.3f},{epoch_uciqe:.3f},{best_uciqe:.3f}"
        )
        with open(log_file, 'a+', encoding='utf-8') as f:
            f.write(log_line + "\n")

        # 保存断点模型（每个epoch都保存，方便断点续训）
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1
        }
        torch.save(checkpoint, os.path.join(save_dir, 'net_checkpoint.pth'))

