## Introduction
&emsp;&emsp;WaveletFusionNet is a novel end-to-end underwater image enhancement network that integrates wavelet decomposition, a transformer-embedded U-Net, and multi-layer residual CBAM modules. First, the input image is decomposed into low- and high-frequency subbands using discrete wavelet transform. The low-frequency subband is processed with a transformer-enhanced U-Net to recover global structure and correct color, while the high-frequency subbands are refined with multi-layer residual CBAM modules to enhance detail and texture. The enhanced subbands are then fused via inverse wavelet transform to reconstruct the final output. Extensive experiments on UIEB and LSUI datasets show that WaveletFusionNet achieves superior results in terms of color correction, detail preservation, and computational efficiency, surpassing previous state-of-the-art approaches.

## Directory Structure
```text
WaveletFusionNet-main/
├── data/                       # Datasets
│   ├── LSUI/                   # LSUI dataset
│   │   ├── test/               # test set
│   │   ├── train/              # train set
│   │   └── Validation/         # validation set
│   ├── U45/                    # U45 dataset (images directly here)
│   └── UIEB/                   # UIEB dataset
│       ├── test/               # test set
│       ├── train/              # train set
│       └── Validation/         # validation set
│
├── Experiments/                # Training & testing scripts
│   ├── test.py                 # Test script
│   └── train.py                # Training script
│
├── Loss/                       # Loss function modules
│   ├── __init__.py
│   └── JointLoss.py            # Joint loss implementation
│
├── models/                     # Saved model weights & structures
│   ├── model_LSUI/             # Models for LSUI dataset
│   └── model_UIEB/             # Models for UIEB dataset
│
├── Net/                        # Network architectures
│   ├── __init__.py
│   ├── HFAB.py                 # High-frequency attention block
│   ├── LFTU.py                 # Low Frequency Transformer U-Net
│   └── main.py                 # Main network definition
│
├── results/                    # Results (inference/visualization etc.)
│
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── data_RGB.py             # Data loading & preprocessing
│   ├── UIQM_UCIQE.py           # Image quality metrics
│   ├── unet.py                 # UNet implementation
│   ├── utils.py                # Common utilities
│   └── wavelet.py              # Wavelet transforms & tools
│
├── requirements.txt            # Python requirements
├── vgg16-397923af.pth          # VGG16 pretrained weights (for perceptual loss)

