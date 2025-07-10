import torch
import torch.nn as nn
import torch.nn.functional as F

"""
实现了一个基于 U-Net 结构的网络模块，包含了一系列的卷积、下采样、上采样和输出操作
"""
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        # 定义一个包含两个连续卷积块的序列
        # 每个卷积块由卷积层、批归一化层和ReLU激活函数组成
        self.conv = nn.Sequential(
            # 第一个卷积层，输入通道数为in_ch，输出通道数为out_ch，卷积核大小为3x3，填充为1
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            # 批归一化层，对out_ch个通道进行归一化
            nn.BatchNorm2d(out_ch),
            # ReLU激活函数，设置inplace=True表示在原张量上进行操作，节省内存
            nn.ReLU(inplace=True),
            # 第二个卷积层，输入通道数为out_ch，输出通道数也为out_ch，卷积核大小为3x3，填充为1
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            # 批归一化层，对out_ch个通道进行归一化
            nn.BatchNorm2d(out_ch),
            # ReLU激活函数，设置inplace=True表示在原张量上进行操作，节省内存
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 将输入x传入定义好的卷积序列中进行计算
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        # 使用double_conv模块作为输入层的卷积操作
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        # 将输入x传入定义好的double_conv模块中进行计算
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        # 定义下采样模块，包含一个最大池化层和一个double_conv模块
        self.mpconv = nn.Sequential(
            # 最大池化层，池化核大小为2x2，步长为2，用于下采样
            nn.MaxPool2d(2),
            # double_conv模块，用于对下采样后的特征进行卷积操作
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        # 将输入x传入定义好的下采样模块中进行计算
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        # 如果使用双线性插值上采样
        if bilinear:
            # 使用双线性插值进行上采样，上采样因子为2，模式为bilinear，设置align_corners=True
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # 使用转置卷积进行上采样，输入通道数为in_ch//2，输出通道数为in_ch//2，卷积核大小为2x2，步长为2
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        # 定义一个double_conv模块，用于对上采样后的特征进行卷积操作
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        # 对输入x1进行上采样操作
        x1 = self.up(x1)

        # 计算x2和上采样后的x1在高度和宽度上的差值
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # 对x1进行填充，使其与x2的尺寸相同
        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        # 沿通道维度（dim=1）将x2和填充后的x1进行拼接
        x = torch.cat([x2, x1], dim=1)
        # 将拼接后的特征传入定义好的double_conv模块中进行计算
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        # 定义输出层的卷积操作，输入通道数为in_ch，输出通道数为out_ch，卷积核大小为1x1
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        # 将输入x传入定义好的卷积层中进行计算
        x = self.conv(x)
        return x
