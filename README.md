# WaveletFusionNet
A PyTorch-based neural network for multi-source image fusion using wavelet transforms.


# Directory Structure
WaveletFusionNet-main/
├── data/                       # Datasets
│   ├── LSUI/                   # LSUI dataset
│   │   ├── test/               # test set
│   │   └── train/              # rain set
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
│   ├── LFTU.py                 # Low-frequency Transformer UNet
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
├── vgg16-397923af.pth          # VGG16 pretrained weights (for percept
