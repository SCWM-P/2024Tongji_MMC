import os
from . import set_figure
from . import img_process
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def WSNR_visual(img1, img2, file_path=None):
    files = {
        '经过滤波的原图':plt.figure(dpi=300),
        '经过滤波的失真图':plt.figure(dpi=300),
        '误差图像':plt.figure(dpi=300),
    }
    wsnr_out = img_process.WSNR(img1, img2)[1:]
    files = {
        title: set_figure(fig, title, content)
        for fig, title, content in
        zip(files.values(), files.keys(), wsnr_out)
    }
    if file_path is not None:
        os.makedirs(os.path.join(file_path, 'WSNR'), exist_ok=True)
        file_path = os.path.join(file_path, 'WSNR')
        for title, fig in files.items():
            fig.savefig(os.path.join(file_path, f'{title}.png'))
        print(f"WSNR processing done! Files are saved in {file_path}")
    return files
