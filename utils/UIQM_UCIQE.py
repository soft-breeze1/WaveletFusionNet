import cv2
import numpy as np
from skimage import transform
from scipy import ndimage


def calculate_uiqm(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """计算UIQM(Unified Image Quality Matrix)图像质量指标"""
    x = img
    x = x.astype(np.float32)
    # UIQM三个分量的权重系数
    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753
    # 计算三个质量分量
    uicm = _uicm(x)  # 颜色度
    uism = _uism(x)  # 清晰度
    uiconm = _uiconm(x)  # 对比度
    # 加权组合三个分量得到最终UIQM值
    uiqm = (c1 * uicm) + (c2 * uism) + (c3 * uiconm)
    return uiqm


def calculate_uciqe(img, img2=None, crop_border=0, input_order='HWC', test_y_channel=False, **kwargs):
    """
    计算UCIQE指标
    img: 输入图像（BGR格式，8位）
    """
    # 添加epsilon以避免除零
    epsilon = 1e-8

    # 转到LAB颜色空间
    img_bgr = img.copy()
    # 确保像素范围在0-255
    # 转为float类型，避免整数相除导致的精度问题
    img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    # 提取L、a、b通道
    img_lum = img_lab[..., 0] / 255.0  # 归一化到[0,1]
    img_a = img_lab[..., 1] / 255.0
    img_b = img_lab[..., 2] / 255.0

    # 计算色度（Chroma）
    img_chr = np.sqrt(np.square(img_a) + np.square(img_b))
    # 计算饱和度
    denominator = np.sqrt(np.square(img_chr) + np.square(img_lum)) + epsilon
    img_sat = img_chr / denominator
    # 处理nan和inf
    img_sat = np.nan_to_num(img_sat, nan=0.0, posinf=0.0, neginf=0.0)

    # 计算色度的平均
    aver_chr = np.mean(img_chr)
    # 计算色度变异
    var_chr = np.sqrt(np.mean(np.abs(1 - np.square(aver_chr / (img_chr + epsilon)))))
    # 某些情况下可能出现负值或nan，转为零
    var_chr = np.nan_to_num(var_chr, nan=0.0, posinf=0.0, neginf=0.0)

    # 计算亮度的直方图和对比度
    # 先裁剪边界
    if crop_border > 0:
        img_lum_cropped = img_lum[crop_border:-crop_border, crop_border:-crop_border]
    else:
        img_lum_cropped = img_lum

    # 直方图和累积分布函数
    dtype = img_lum_cropped.dtype
    nbins = 256 if dtype != np.uint16 else 65536
    hist, bins = np.histogram(img_lum_cropped.flatten(), bins=nbins, range=(0,1))
    cdf = np.cumsum(hist) / np.sum(hist)
    ilow = np.where(cdf > 0.0100)[0]
    ihigh = np.where(cdf >= 0.9900)[0]
    if len(ilow) > 0 and len(ihigh) > 0:
        tol = [ilow[0] / nbins, ihigh[0] / nbins]
        con_lum = tol[1] - tol[0]
    else:
        con_lum = 0

    # 计算对比度
    # 设置系数
    alpha1, alpha2, alpha3 = 0.4680, 0.2745, 0.2576
    quality_score = alpha1 * var_chr + alpha2 * con_lum + alpha3 * np.mean(img_sat)
    return quality_score


def _uicm(img):
    """计算UICM(Uniform Colorfulness Measure)颜色度指标"""
    img = np.array(img, dtype=np.float64)
    # 分离RGB通道
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # 计算红绿(RG)和黄蓝(YB)颜色差异
    RG = R - G
    YB = (R + G) / 2 - B

    # 计算图像像素总数
    K = R.shape[0] * R.shape[1]

    # 处理RG通道
    RG1 = RG.reshape(1, K)
    RG1 = np.sort(RG1)
    # 移除两端10%的极端值
    alphaL = 0.1
    alphaR = 0.1
    RG1 = RG1[0, int(alphaL * K + 1):int(K * (1 - alphaR))]
    N = K * (1 - alphaR - alphaL)
    # 计算均值和标准差
    meanRG = np.sum(RG1) / N
    deltaRG = np.sqrt(np.sum((RG1 - meanRG) ** 2) / N)

    # 处理YB通道，步骤与RG通道相同
    YB1 = YB.reshape(1, K)
    YB1 = np.sort(YB1)
    YB1 = YB1[0, int(alphaL * K + 1):int(K * (1 - alphaR))]
    N = K * (1 - alphaR - alphaL)
    meanYB = np.sum(YB1) / N
    deltaYB = np.sqrt(np.sum((YB1 - meanYB) ** 2) / N)

    # 组合RG和YB的统计特性得到UICM值
    uicm = -0.0268 * np.sqrt(meanRG ** 2 + meanYB ** 2) + 0.1586 * np.sqrt(deltaYB ** 2 + deltaRG ** 2)
    return uicm


def _uiconm(img):
    """计算UICONM(Uniform Contrast Measure)对比度指标"""
    img = np.array(img, dtype=np.float64)
    # 分离RGB通道
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # 定义分块大小
    patchez = 5
    m = R.shape[0]
    n = R.shape[1]

    # 如果图像尺寸不是patchez的整数倍，则调整图像大小
    if m % patchez != 0 or n % patchez != 0:
        x = int(m - m % patchez + patchez)
        y = int(n - n % patchez + patchez)
        R = transform.resize(R, (x, y))
        G = transform.resize(G, (x, y))
        B = transform.resize(B, (x, y))

    m = R.shape[0]
    n = R.shape[1]
    # 计算分块数量
    k1 = m / patchez
    k2 = n / patchez

    # 计算R通道的AMEE(Advanced Mean Edge Energy)
    AMEER = 0
    for i in range(0, m, patchez):
        for j in range(0, n, patchez):
            sz = patchez
            im = R[i:i + sz, j:j + sz]
            Max = np.max(im)
            Min = np.min(im)
            # 避免除零和无效对数计算
            if (Max != 0 or Min != 0) and Max != Min:
                AMEER = AMEER + np.log((Max - Min) / (Max + Min)) * ((Max - Min) / (Max + Min))
    AMEER = 1 / (k1 * k2) * np.abs(AMEER)

    # 计算G通道的AMEE，步骤与R通道相同
    AMEEG = 0
    for i in range(0, m, patchez):
        for j in range(0, n, patchez):
            sz = patchez
            im = G[i:i + sz, j:j + sz]
            Max = np.max(im)
            Min = np.min(im)
            if (Max != 0 or Min != 0) and Max != Min:
                AMEEG = AMEEG + np.log((Max - Min) / (Max + Min)) * ((Max - Min) / (Max + Min))
    AMEEG = 1 / (k1 * k2) * np.abs(AMEEG)

    # 计算B通道的AMEE，步骤与R通道相同
    AMEEB = 0
    for i in range(0, m, patchez):
        for j in range(0, n, patchez):
            sz = patchez
            im = B[i:i + sz, j:j + sz]
            Max = np.max(im)
            Min = np.min(im)
            if (Max != 0 or Min != 0) and Max != Min:
                AMEEB = AMEEB + np.log((Max - Min) / (Max + Min)) * ((Max - Min) / (Max + Min))
    AMEEB = 1 / (k1 * k2) * np.abs(AMEEB)

    # 组合三个通道的AMEE值得到UICONM
    uiconm = AMEER + AMEEG + AMEEB
    return uiconm


def _uism(img):
    """计算UISM(Uniform Image Sharpness Measure)清晰度指标"""
    img = np.array(img, dtype=np.float64)
    # 分离RGB通道
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # 定义Sobel算子
    hx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # x方向梯度算子
    hy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # y方向梯度算子

    # 计算各通道的Sobel边缘响应
    SobelR = np.abs(ndimage.convolve(R, hx, mode='nearest') + ndimage.convolve(R, hy, mode='nearest'))
    SobelG = np.abs(ndimage.convolve(G, hx, mode='nearest') + ndimage.convolve(G, hy, mode='nearest'))
    SobelB = np.abs(ndimage.convolve(B, hx, mode='nearest') + ndimage.convolve(B, hy, mode='nearest'))

    # 定义分块大小
    patchez = 5
    m = R.shape[0]
    n = R.shape[1]

    # 如果图像尺寸不是patchez的整数倍，则调整图像大小
    if m % patchez != 0 or n % patchez != 0:
        x = int(m - m % patchez + patchez)
        y = int(n - n % patchez + patchez)
        SobelR = transform.resize(SobelR, (x, y))
        SobelG = transform.resize(SobelG, (x, y))
        SobelB = transform.resize(SobelB, (x, y))

    m = SobelR.shape[0]
    n = SobelR.shape[1]
    # 计算分块数量
    k1 = m / patchez
    k2 = n / patchez

    # 计算R通道的EME(Edge-Magnitude Evaluation)
    EMER = 0
    for i in range(0, m, patchez):
        for j in range(0, n, patchez):
            sz = patchez
            im = SobelR[i:i + sz, j:j + sz]
            Max = np.max(im)
            Min = np.min(im)
            # 避免除零和无效对数计算
            if Max != 0 and Min != 0:
                EMER = EMER + np.log(Max / Min)
    EMER = 2 / (k1 * k2) * np.abs(EMER)

    # 计算G通道的EME，步骤与R通道相同
    EMEG = 0
    for i in range(0, m, patchez):
        for j in range(0, n, patchez):
            sz = patchez
            im = SobelG[i:i + sz, j:j + sz]
            Max = np.max(im)
            Min = np.min(im)
            if Max != 0 and Min != 0:
                EMEG = EMEG + np.log(Max / Min)
    EMEG = 2 / (k1 * k2) * np.abs(EMEG)

    # 计算B通道的EME，步骤与R通道相同
    EMEB = 0
    for i in range(0, m, patchez):
        for j in range(0, n, patchez):
            sz = patchez
            im = SobelB[i:i + sz, j:j + sz]
            Max = np.max(im)
            Min = np.min(im)
            if Max != 0 and Min != 0:
                EMEB = EMEB + np.log(Max / Min)
    EMEB = 2 / (k1 * k2) * np.abs(EMEB)

    # 根据RGB重要性权重组合三个通道的EME值
    lambdaR = 0.299  # R通道权重
    lambdaG = 0.587  # G通道权重(最高，因为人眼对绿色最敏感)
    lambdaB = 0.114  # B通道权重
    uism = lambdaR * EMER + lambdaG * EMEG + lambdaB * EMEB
    return uism

# if __name__ == '__main__':
#     # 读取测试图像（建议使用公开数据集图像或自定义图像）
#     img_path = r"D:\EI\Net-1\data\UIEB\test\input\918_img_.png"  # 替换为实际图像路径
#     img = cv2.imread(img_path)
#
#     # 测试 UIQM
#     uiqm_score = calculate_uiqm(img, img2=None, crop_border=0)
#     print(f"UIQM 得分: {uiqm_score:.4f}")
#
#     # 测试 UCIQE
#     uciqe_score = calculate_uciqe(img, img2=None, crop_border=0)
#     print(f"UCIQE 得分: {uciqe_score:.4f}")