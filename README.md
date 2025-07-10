## WaveletFusionNet
    WaveletFusionNet is a novel end-to-end underwater image enhancement network that integrates wavelet decomposition, a transformer-embedded U-Net, and multi-layer residual CBAM modules.First, the input image is decomposed into low- and high-frequency subbands using discrete wavelet transform. The low-frequency subband is processed with a transformer-enhanced U-Net to recover global structure and correct color, while the high-frequency subbands are refined with multi-layer residual CBAM modules to enhance detail and texture. The enhanced subbands are then fused via inverse wavelet transform to reconstruct the final output.Extensive experiments on UIEB and LSUI datasets show that WaveletFusionNet achieves superior results in terms of color correction, detail preservation, and computational efficiency, surpassing previous state-of-the-art approaches.

## Directory Structure
```text
WaveletFusionNet-main/
├── data/                       # 数据集文件夹 / Datasets
│   ├── LSUI/                   # LSUI 数据集 / LSUI dataset
│   │   ├── test/               # 测试集 / test set
│   │   └── train/              # 训练集 / train set
|   |   └── Validation/         # 验证集 / validation set
│   ├── U45/                    # U45 数据集（图片直接存放于此，无子目录）/ U45 dataset (images directly here)
│   └── UIEB/                   # UIEB 数据集 / UIEB dataset
│       ├── test/               # 测试集 / test set
│       ├── train/              # 训练集 / train set
│       └── Validation/         # 验证集 / validation set
│
├── Experiments/                # 训练和测试脚本 / Training & testing scripts
│   ├── test.py                 # 测试脚本 / Test script
│   └── train.py                # 训练脚本 / Training script
│
├── Loss/                       # 损失函数模块 / Loss function modules
│   ├── __init__.py
│   └── JointLoss.py            # 联合损失实现 / Joint loss implementation
│
├── models/                     # 保存模型参数及结构 / Saved model weights & structures
│   ├── model_LSUI/             # LSUI 数据集模型 / Models for LSUI dataset
│   └── model_UIEB/             # UIEB 数据集模型 / Models for UIEB dataset
│
├── Net/                        # 网络结构定义 / Network architectures
│   ├── __init__.py
│   ├── HFAB.py                 # 高频注意力块 / High-frequency attention block
│   ├── LFTU.py                 # 低频Transformer UNet / Low Frequency Transformer U-Net
│   └── main.py                 # 网络主结构入口 / Main network definition
│
├── results/                    # 结果保存（推理/可视化等）/ Results (inference/visualization etc.)
│
├── utils/                      # 工具函数 / Utilities
│   ├── __init__.py
│   ├── data_RGB.py             # 数据加载与预处理 / Data loading & preprocessing
│   ├── UIQM_UCIQE.py           # 图像质量评价指标 / Image quality metrics
│   ├── unet.py                 # UNet 相关实现 / UNet implementation
│   ├── utils.py                # 通用工具 / Common utilities
│   └── wavelet.py              # 小波相关操作 / Wavelet transforms & tools
│
├── requirements.txt            # Python依赖列表 / Python requirements
├── vgg16-397923af.pth          # VGG16 预训练权重（用于感知损失等）/ VGG16 pretrained weights (for perceptual loss)
