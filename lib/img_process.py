import cv2
import numpy as np
from numpy.fft import fft2, fftshift
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from skimage.metrics import structural_similarity as ssim


def FSIMc(image_distorted, image_reference):
    def _gradient_similarity(grad1, grad2):
        magnitude_similarity = (2 * np.abs(grad1) * np.abs(grad2) + 1e-15) / (
                    np.abs(grad1) ** 2 + np.abs(grad2) ** 2 + 1e-15)
        phase_consistency = (1 - np.abs(np.angle(grad1 / (grad2 + 1e-15)) / np.pi)) ** 2
        return magnitude_similarity, phase_consistency

    def _compute_channel_similarity(channel1, channel2):
        grad1 = cv2.Sobel(channel1, cv2.CV_64F, 1, 0, ksize=5) + 1j * cv2.Sobel(channel1, cv2.CV_64F, 0, 1, ksize=5)
        grad2 = cv2.Sobel(channel2, cv2.CV_64F, 1, 0, ksize=5) + 1j * cv2.Sobel(channel2, cv2.CV_64F, 0, 1, ksize=5)
        mag_sim, phase_con = _gradient_similarity(grad1, grad2)
        return np.sum(mag_sim * phase_con) / np.sum(phase_con)

    # 分别计算 R, G, B 通道的相似度
    r_similarity = _compute_channel_similarity(image_distorted[:, :, 2], image_reference[:, :, 2])
    g_similarity = _compute_channel_similarity(image_distorted[:, :, 1], image_reference[:, :, 1])
    b_similarity = _compute_channel_similarity(image_distorted[:, :, 0], image_reference[:, :, 0])

    # 计算总的 FSIMc 值，这里简单地取平均
    fsim_c = (r_similarity + g_similarity + b_similarity) / 3
    return fsim_c,


def SSIM(image_distorted, image_reference):
    gray_image1 = cv2.cvtColor(image_distorted, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image_reference, cv2.COLOR_BGR2GRAY)
    ssim_score = ssim(
        gray_image1, gray_image2,
        data_range=max(
            gray_image2.max(),
            gray_image1.max()  
        ) - min(
            gray_image2.min(),
            gray_image1.min()
        )
    )
    return ssim_score,


def MSSIM(image_distorted, image_reference, K=None, window=None, L=255):
    img1 = cv2.cvtColor(image_distorted, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image_reference, cv2.COLOR_BGR2GRAY)
    # 检查图像是否相同大小
    if img1.shape != img2.shape:
        raise ValueError('Images must have the same dimensions.')
    if K is None:
        K = [0.01, 0.03]
    if window is None:
        window = cv2.getGaussianKernel(11, 1.5) * cv2.getGaussianKernel(11, 1.5).T
    # 计算 SSIM
    C1 = (K[0] ** 2) * L
    C2 = (K[1] ** 2) * L
    mean1 = cv2.filter2D(img1, -1, window, borderType=cv2.BORDER_REFLECT)
    mean2 = cv2.filter2D(img2, -1, window, borderType=cv2.BORDER_REFLECT)
    mean1_sq = mean1 ** 2
    mean2_sq = mean2 ** 2
    mean12 = mean1 * mean2
    var1 = cv2.filter2D(img1 ** 2, -1, window, borderType=cv2.BORDER_REFLECT) - mean1_sq
    var2 = cv2.filter2D(img2 ** 2, -1, window, borderType=cv2.BORDER_REFLECT) - mean2_sq
    covar12 = cv2.filter2D(img1 * img2, -1, window, borderType=cv2.BORDER_REFLECT) - mean12
    ssim_map = ((2 * mean12 + C1) * (2 * covar12 + C2)) / ((mean1_sq + mean2_sq + C1) * (var1 + var2 + C2))
    mssim = np.mean(ssim_map)
    return mssim, ssim_map, mean1, mean2, mean12, var1, var2, covar12


def PSNR(image_distorted, image_reference):
    def _mse(imageA, imageB):
        # 计算两个图像的均方误差
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])
        return err
    # 计算峰值信噪比
    mse_value = _mse(image_distorted, image_reference)
    if mse_value == 0:
        return 100
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse_value))
    return psnr_value,


def PSNRc(image_distorted, image_reference):
    img1_YCbCr = cv2.cvtColor(image_distorted, cv2.COLOR_BGR2YCrCb)
    img2_YCbCr = cv2.cvtColor(image_reference, cv2.COLOR_BGR2YCrCb)

    img1_Y, _, _ = cv2.split(img1_YCbCr)
    img2_Y, _, _ = cv2.split(img2_YCbCr)

    mse = np.mean((img1_Y.astype(np.float64) - img2_Y.astype(np.float64)) ** 2)
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr, img1_Y, img2_Y


def NSS(*image):
    def calculate_lbp(image):
        # 计算LBP特征
        lbp_image = local_binary_pattern(image, 8, 1, method='uniform')
        hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, 10), range=(0, 9))
        lbp_mean = np.mean(hist)
        lbp_std_dev = np.std(hist)
        return lbp_mean, lbp_std_dev, lbp_image

    def calculate_hog(image):
        # 计算HOG特征
        hog_features_r = hog(
            image[:, :, 0], orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            block_norm='L2-Hys'
        )
        hog_features_g = hog(
            image[:, :, 1], orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            block_norm='L2-Hys'
        )
        hog_features_b = hog(
            image[:, :, 2], orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            block_norm='L2-Hys'
        )
        hog_features = np.concatenate((hog_features_r, hog_features_g, hog_features_b))
        hog_mean = np.mean(hog_features)
        hog_std_dev = np.std(hog_features)
        return hog_mean, hog_std_dev, hog_features

    def calculate_nss_features(image):
        # 计算图像统计量
        mean_value = np.mean(image)
        std_dev = np.std(image)

        # 计算图像梯度
        dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(dx ** 2 + dy ** 2)
        gradient_mean = np.mean(gradient_magnitude)
        gradient_std_dev = np.std(gradient_magnitude)

        # 计算图像直方图
        hist = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
        hist_mean = np.mean(hist)
        hist_std_dev = np.std(hist)

        # 计算灰度共生矩阵特征
        glcm = graycomatrix(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), [1], [0], 256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')
        energy = graycoprops(glcm, 'energy')
        contrast_mean = np.mean(contrast)
        energy_mean = np.mean(energy)

        # 计算LBP特征
        lbp_mean, lbp_std_dev, lbp_image = calculate_lbp(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        # 计算HOG特征
        hog_mean, hog_std_dev, hog_features = calculate_hog(image)

        return (
            mean_value, std_dev,
            gradient_mean, gradient_std_dev,
            hist_mean, hist_std_dev,
            contrast_mean, energy_mean,
            lbp_mean, lbp_std_dev,
            hog_mean, hog_std_dev
        ), lbp_image, hog_features, gradient_magnitude, hist, glcm

    image = image[0] if isinstance(image, tuple) else image
    if image is None:
        print("未找到图像！")
        return None
    # 计算NSS特征
    features, lbp_image, hog_features, gradient_magnitude, hist, glcm = calculate_nss_features(image)
    # 计算质量指标
    quality_index = sum(features) / len(features)

    return quality_index, features, lbp_image, hog_features, gradient_magnitude, hist, glcm


def WSNR(image_distorted, image_reference):
    def cal_wsnr(original_image, distorted_image, csf):
        # 将图像转换为浮点型以进行计算
        original_image = original_image.astype(np.float32) / 255.0
        distorted_image = distorted_image.astype(np.float32) / 255.0
        # 应用高斯低通滤波器（CSF）
        original_image_filtered = cv2.filter2D(original_image, -1, csf)
        distorted_image_filtered = cv2.filter2D(distorted_image, -1, csf)
        # 计算误差图像
        error_image = original_image_filtered - distorted_image_filtered
        # 计算误差图像的FFT
        error_fft = fft2(fftshift(error_image))
        # 计算加权误差图像的功率
        weighted_error_power = np.sum(np.abs(error_fft) ** 2)
        # 计算WSNR
        wsnr = 10 * np.log10(np.sum(np.abs(fft2(fftshift(original_image_filtered))) ** 2) / weighted_error_power)

        return wsnr, original_image_filtered, distorted_image_filtered, error_image

    def cal_csf(image_shape, sigma=1.2):
        # 创建一个高斯低通滤波器，用于模拟CSF
        ax = np.arange(-image_shape[0] // 2, image_shape[0] // 2 + 1)
        ay = np.arange(-image_shape[1] // 2, image_shape[1] // 2 + 1)
        xx, yy = np.meshgrid(ax, ay)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel /= np.sum(kernel)
        return kernel

    csf = cal_csf(image_reference.shape, sigma=1.2)
    wsnr_value, original_image_filtered, distorted_image_filtered, error_image = cal_wsnr(image_reference, image_distorted, csf)
    return wsnr_value, original_image_filtered, distorted_image_filtered, error_image

