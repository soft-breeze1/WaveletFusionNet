import argparse
import os
import time
import math
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

# 自定义模块导入
from Net.main import WaveletFusionNet
from utils.data_RGB import get_validation_data
from utils.utils import batch_PSNR, batch_SSIM
from utils.UIQM_UCIQE import calculate_uciqe, calculate_uiqm  # 引入统一实现的指标

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='WaveletFusionNet Testing')
# 硬件参数
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--device', type=str, default='cuda:0')
# 数据参数
parser.add_argument('--test_dir', type=str, default=r'D:\EI\Net-1\data\UIEBD\test', help='测试数据根目录')
parser.add_argument('--testBatchSize', type=int, default=1, help='测试批次大小')
parser.add_argument('--threads', type=int, default=4, help='数据加载线程数')
# 模型参数
parser.add_argument('--model', type=str, default=r'D:\EI\Net-2\model_all\net_checkpoint.pth', help='预训练模型路径')
# 输出参数
parser.add_argument('--save_dir', type=str, default=r'D:\EI\Net-1\results', help='结果保存目录')
parser.add_argument('--save_visual', default=True, help='是否保存可视化结果')

# 解析参数
opt = parser.parse_args()

# 设备设置
device = torch.device(opt.device if opt.gpu_mode and torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


def main():
    # 创建保存目录
    os.makedirs(opt.save_dir, exist_ok=True)

    # 加载测试数据集
    print('===> Loading test dataset')
    test_set = get_validation_data(opt.test_dir, {'patch_size': None})  # 不进行裁剪
    test_loader = DataLoader(
        test_set,
        batch_size=opt.testBatchSize,
        shuffle=False,
        num_workers=opt.threads
    )

    # 初始化模型
    print('===> Building model')
    model = WaveletFusionNet().to(device)

    # 加载预训练权重
    if os.path.exists(opt.model):
        checkpoint = torch.load(opt.model, map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print(f'Loaded model from {opt.model}')
    else:
        raise FileNotFoundError(f"Model file not found: {opt.model}")

    # 评估模式
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_uciqe = 0.0
    total_uiqm = 0.0
    timer = 0.0

    with torch.no_grad():
        for batch_idx, (target, input_, filenames) in enumerate(test_loader):
            # 数据迁移到设备
            input_tensor = input_.to(device)
            target_tensor = target.to(device) if target is not None else None

            # 推理计时
            start = time.time()
            output = model(input_tensor)
            elapsed = time.time() - start
            timer += elapsed

            # 后处理
            output = torch.clamp(output, 0.0, 1.0)

            # 计算指标
            if target_tensor is not None:
                psnr_val = batch_PSNR(output, target_tensor, 1.0)
                ssim_val = batch_SSIM(output, target_tensor, 1.0)
                total_psnr += psnr_val
                total_ssim += ssim_val

            # 计算无参考指标
            batch_uciqe = 0.0
            batch_uiqm = 0.0
            for idx in range(output.size(0)):
                # 转换为OpenCV格式
                img = output[idx].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 转换为BGR格式

                # 使用统一函数计算指标
                batch_uciqe += calculate_uciqe(img_bgr)
                batch_uiqm += calculate_uiqm(img_bgr, img2=None, crop_border=0, input_order='HWC')

            total_uciqe += batch_uciqe / output.size(0)
            total_uiqm += batch_uiqm / output.size(0)

            # 保存结果
            if opt.save_visual:
                save_images(output, filenames, opt.save_dir)

            # 打印进度
            print(f'Processed {batch_idx + 1}/{len(test_loader)} | '
                  f'Time: {elapsed:.4f}s | '
                  f'PSNR: {psnr_val:.2f} | '
                  f'SSIM: {ssim_val:.4f} | '
                  f'UCIQE: {batch_uciqe / output.size(0):.3f} | '
                  f'UIQM: {batch_uiqm / output.size(0):.3f}')

    # 打印最终统计结果
    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    avg_uciqe = total_uciqe / len(test_loader)
    avg_uiqm = total_uiqm / len(test_loader)

    print('\n=== Final Results ===')
    print(f'Total images: {len(test_loader)}')
    print(f'Average PSNR: {avg_psnr:.2f} dB')
    print(f'Average SSIM: {avg_ssim:.4f}')
    print(f'Average UCIQE: {avg_uciqe:.3f}')
    print(f'Average UIQM: {avg_uiqm:.3f}')
    print(f'Average time: {timer / len(test_loader):.4f}s per image')

    # 结果保存到文本文件
    results_file = os.path.join(opt.save_dir, 'results_all.txt')
    with open(results_file, 'a') as f:  # 使用追加模式，方便多次测试累积
        f.write(f"Test on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total images: {len(test_loader)}\n")
        f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Average UCIQE: {avg_uciqe:.3f}\n")
        f.write(f"Average UIQM: {avg_uiqm:.3f}\n")
        f.write(f"Average time: {timer / len(test_loader):.4f}s per image\n")
        f.write('-' * 40 + '\n')

def save_images(tensor, filenames, save_dir):
    """保存张量为图像文件"""
    for idx in range(tensor.size(0)):
        # 张量转numpy
        img = tensor[idx].cpu().numpy().transpose(1, 2, 0)

        # 转换到0-255范围
        img = (img * 255).clip(0, 255).astype('uint8')

        # BGR转RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 构建保存路径，确保带有合法扩展名
        base_name = os.path.splitext(os.path.basename(filenames[idx]))[0]
        save_path = os.path.join(save_dir, base_name + '.png')

        # 保存图像
        cv2.imwrite(save_path, img)


if __name__ == '__main__':
    main()
