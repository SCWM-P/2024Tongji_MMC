import os
from . import set_figure
from . import img_process
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def MSSIM_visual(img1, img2, file_path=None):
    files = {
        'SSIM 图': plt.figure(dpi=300),
        '原图的均值': plt.figure(dpi=300),
        '失真图的均值': plt.figure(dpi=300),
        '原图与失真图乘积的均值': plt.figure(dpi=300),
        '原图的方差': plt.figure(dpi=300),
        '失真图的方差': plt.figure(dpi=300),
        '原图与失真图的协方差': plt.figure(dpi=300),
    }
    ssim_output = img_process.MSSIM(img1, img2)[1:]
    files = {
        title: set_figure(fig, title, content)
        for fig, title, content in
        zip(files.values(), files.keys(), ssim_output)
    }
    if file_path is not None:
        os.makedirs(os.path.join(file_path, 'MSSIM'), exist_ok=True)
        file_path = os.path.join(file_path, 'MSSIM')
        for title, fig in files.items():
            fig.savefig(os.path.join(file_path, f'MSSIM_{title}.png'), transparent=True)
        print(f"MMSIM processing done! Files are saved in {file_path}")

    return files
