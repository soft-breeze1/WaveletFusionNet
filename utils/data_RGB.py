import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2


class RGBDataset(Dataset):
    """支持子目录的RGB图像数据集类，处理input和target配对图像"""

    def __init__(self, data_dir, patch_size=None, augment=True):
        """
        初始化数据集

        参数:
            data_dir: 数据根目录（包含input和target子目录）
            patch_size: 训练时裁剪的patch大小，验证时可设为None
            augment: 是否进行数据增强
        """
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.augment = augment

        # 分别获取input和target目录路径
        self.input_dir = os.path.join(data_dir, 'input')
        self.target_dir = os.path.join(data_dir, 'target')

        # 检查目录是否存在
        if not (os.path.exists(self.input_dir) and os.path.exists(self.target_dir)):
            raise ValueError(f"缺少input或target子目录，路径：{data_dir}")

        # 查找配对的图像文件（假设文件名相同）
        self.image_files = self._find_paired_files()

        # 定义图像转换
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为Tensor，范围[0,1]
        ])

        # 数据增强参数
        self.angles = [0, 90, 180, 270]
        self.flips = [0, 1, -1]  # 0: 水平翻转, 1: 垂直翻转, -1: 水平和垂直翻转

    def _find_paired_files(self):
        """查找input和target中配对的图像文件（文件名需一致）"""
        input_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
        target_extensions = input_extensions  # 目标图像扩展名应与输入一致

        # 获取input和target的文件名集合
        input_files = {os.path.splitext(f)[0] for f in os.listdir(self.input_dir)
                       if os.path.splitext(f)[1].lower() in input_extensions}
        target_files = {os.path.splitext(f)[0] for f in os.listdir(self.target_dir)
                        if os.path.splitext(f)[1].lower() in target_extensions}

        # 取交集作为有效配对文件
        common_files = input_files & target_files
        return list(common_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """获取配对的输入-目标图像样本"""
        filename = self.image_files[idx]

        # 加载输入和目标图像
        input_path = os.path.join(self.input_dir, f"{filename}{self._get_extension(self.input_dir, filename)}")
        target_path = os.path.join(self.target_dir, f"{filename}{self._get_extension(self.target_dir, filename)}")

        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')

        # 转换为numpy数组
        input_np = np.array(input_img)
        target_np = np.array(target_img)

        input_np, target_np = self._make_hw_even(input_np, target_np)

        # 统一裁剪（训练时随机裁剪，验证时中心裁剪或不裁剪）
        if self.patch_size is not None:
            input_np, target_np = self._crop_pair(input_np, target_np, self.patch_size)

        # 数据增强（仅对输入和目标同时操作）
        if self.augment:
            input_np, target_np = self._augment_pair(input_np, target_np)

        # 转换为Tensor
        input_tensor = self.transform(input_np)
        target_tensor = self.transform(target_np)

        return target_tensor, input_tensor, filename  # 返回(target, input, filename)

    # 偶数
    def _make_hw_even(self, input_img, target_img):
        """将H/W裁剪为偶数，防止DWT尺寸不一致错误"""
        h, w, _ = input_img.shape
        h_even = h - (h % 2)
        w_even = w - (w % 2)

        input_img = input_img[:h_even, :w_even, :]
        target_img = target_img[:h_even, :w_even, :]
        return input_img, target_img

    def _get_extension(self, dir_path, filename):
        """获取文件扩展名（处理大小写和多种格式）"""
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            if os.path.exists(os.path.join(dir_path, f"{filename}{ext}")):
                return ext
        raise ValueError(f"未找到文件：{filename} 在 {dir_path} 中")

    def _crop_pair(self, input_img, target_img, patch_size):
        """同时裁剪输入和目标图像"""
        h, w, _ = input_img.shape
        if h < patch_size or w < patch_size:
            # 尺寸不足时缩放
            input_img = cv2.resize(input_img, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
            target_img = cv2.resize(target_img, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
            return input_img, target_img

        # 随机裁剪（训练时）或中心裁剪（验证时）
        if self.augment:
            i = random.randint(0, h - patch_size)
            j = random.randint(0, w - patch_size)
        else:
            # 中心裁剪（验证时使用）
            i = (h - patch_size) // 2
            j = (w - patch_size) // 2

        input_cropped = input_img[i:i + patch_size, j:j + patch_size, :]
        target_cropped = target_img[i:i + patch_size, j:j + patch_size, :]
        return input_cropped, target_cropped

    def _augment_pair(self, input_img, target_img):
        """同时对输入-目标图像进行数据增强"""
        # 随机旋转
        angle = random.choice(self.angles)
        if angle != 0:
            input_img = np.rot90(input_img, angle // 90).copy()
            target_img = np.rot90(target_img, angle // 90).copy()

        # 随机翻转
        flip = random.choice(self.flips)
        if flip != -2:
            input_img = cv2.flip(input_img, flip)
            target_img = cv2.flip(target_img, flip)

        return input_img, target_img


def get_training_data(data_dir, params):
    """获取训练数据集（包含input和target子目录）"""
    patch_size = params.get('patch_size', 256)
    return RGBDataset(data_dir, patch_size=patch_size, augment=True)  # data_dir直接传入train目录

def get_validation_data(data_dir, params):
    """获取验证数据集（包含input和target子目录）"""
    patch_size = params.get('patch_size', None)  # 验证时不裁剪或使用中心裁剪
    return RGBDataset(data_dir, patch_size=patch_size, augment=False)  # data_dir直接传入test目录

# def count_paired_files(data_dir):
#     """统计数据目录中input和target的文件数目及配对情况"""
#     input_dir = os.path.join(data_dir, 'input')
#     target_dir = os.path.join(data_dir, 'target')
#
#     # 获取所有支持的扩展名（与RGBDataset一致）
#     extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
#     ext_set = {ext.lower() for ext in extensions}
#
#     # 统计input文件基名（不带扩展名）
#     input_files = set()
#     for f in os.listdir(input_dir):
#         base, ext = os.path.splitext(f)
#         if ext.lower() in ext_set:
#             input_files.add(base)
#
#     # 统计target文件基名
#     target_files = set()
#     for f in os.listdir(target_dir):
#         base, ext = os.path.splitext(f)
#         if ext.lower() in ext_set:
#             target_files.add(base)
#
#     # 配对数目（交集）
#     common_files = input_files & target_files
#
#     print(f"数据目录: {data_dir}")
#     print(f"input文件总数: {len(input_files)}")
#     print(f"target文件总数: {len(target_files)}")
#     print(f"有效配对数目（同名文件）: {len(common_files)}\n")
#     return len(input_files), len(target_files), len(common_files)
#
#
# if __name__ == '__main__':
#     # 测试训练集
#     train_dir = r'D:\EI\Net\data\UIEB\train'
#     count_paired_files(train_dir)
#
#     # 测试验证集
#     val_dir = r'D:\EI\Net\data\UIEB\test'
#     count_paired_files(val_dir)
