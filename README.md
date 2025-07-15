# Introduction
&emsp;&emsp;WaveletFusionNet is a novel end-to-end underwater image enhancement network that integrates wavelet decomposition, a transformer-embedded U-Net, and multi-layer residual CBAM modules. First, the input image is decomposed into low- and high-frequency subbands using discrete wavelet transform. The low-frequency subband is processed with a transformer-enhanced U-Net to recover global structure and correct color, while the high-frequency subbands are refined with multi-layer residual CBAM modules to enhance detail and texture. The enhanced subbands are then fused via inverse wavelet transform to reconstruct the final output. Extensive experiments on UIEB and LSUI datasets show that WaveletFusionNet achieves superior results in terms of color correction, detail preservation, and computational efficiency, surpassing previous state-of-the-art approaches.

# Directory Structure

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
```

# Environment Setup
- "Python 3.8"
- "PyTorch 2.2.1 (CUDA 12 supported)"
- "torchvision 0.17.1"
- "torchaudio 2.2.1"
- "All dependencies specified in `requirements.txt`"

```bash
# Create environment
conda create -n env_name python=3.8

# Install PyTorch components
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

# Data Preparation
The dataset is provided as a compressed file and can be downloaded from Baidu Netdisk:
- **Download link:** [https://pan.baidu.com/s/1_UZCtZ-KmEMHXkh7GtMzhQ?pwd=0627)

# Pre-trained model
File shared via network disk: models.zip
- **Download link:** [https://pan.baidu.com/s/1RqHZ4rk2jzqRDkpKmkIYpQ?pwd=a2bf)
Extraction code: a2bf

# Train
```bash
python Experiments/train.py
```

# Test
```bash
python Experiments/test.py
```
