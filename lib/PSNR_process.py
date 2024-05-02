import os
import numpy as np
from . import set_figure
from . import img_process
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


img1_path = "C:\\Users\\admin\\Desktop\\tid\\reference_images\\I01.BMP"
img2_path = "C:\\Users\\admin\\Desktop\\tid\\distorted_images\\i01_08_1.bmp"


def PSNR_visual(img1, img2, file_path=None):
    files = {
        '图像1的Y通道': plt.figure(dpi=300),
        '图像2的Y通道': plt.figure(dpi=300),
        'Y通道差异': plt.figure(dpi=300),
    }
    psnrc, img1_Y, img2_Y = img_process.PSNRc(img1, img2)
    psnrc_out = (img1_Y, img2_Y, np.abs(img1_Y - img2_Y))
    files = {
        title: set_figure(fig, title, content)
        for fig, title, content in
        zip(files.values(), files.keys(), psnrc_out)
    }
    if file_path is not None:
        os.makedirs(os.path.join(file_path, 'PSNR'), exist_ok=True)
        file_path = os.path.join(file_path, 'PSNR')
        for title, fig in files.items():
            fig.savefig(os.path.join(file_path, f'{title}.png'), transparent=True)
        print(f"PSNR processing done! Files are saved in {file_path}")
    return files
