import os
from . import img_process
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def NSS_visual(img, file_path=None):
    features, lbp_image, hog_features, gradient_magnitude, hist, glcm = img_process.NSS(img)[1:]
    files = {}

    # 可视化中间量
    files['梯度图像'] = plt.figure(dpi=300)
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title('梯度幅度图像', fontsize=16)
    plt.colorbar()

    files['直方图'] = plt.figure(dpi=300)
    plt.plot(hist)
    plt.title('图像直方图', fontsize=16)
    plt.xlabel('像素强度', fontsize=16)
    plt.ylabel('频率', fontsize=16)

    files['灰度共生矩阵'] = plt.figure(dpi=300)
    plt.imshow(glcm[:, :, 0, 0], cmap='viridis', interpolation='nearest')
    plt.title('灰度共生矩阵', fontsize=16)
    plt.colorbar()

    files['局部二值模式图像'] = plt.figure(dpi=300)
    plt.imshow(lbp_image, cmap='gray')
    plt.title('局部二值模式图像', fontsize=16)
    plt.colorbar()

    files['方向梯度直方图'] = plt.figure(dpi=300)
    plt.plot(hog_features)
    plt.title('方向梯度直方图', fontsize=16)
    plt.xlabel('特征索引', fontsize=16)
    plt.ylabel('特征值', fontsize=16)

    if file_path is not None:
        os.makedirs(os.path.join(file_path, 'NSS'), exist_ok=True)
        file_path = os.path.join(file_path, 'NSS')
        for name, file in files.items():
            file.savefig(os.path.join(file_path, f'NSS_{name}.png'))
        print(f"NSS processing done! Files are saved in {file_path}")

    return files
