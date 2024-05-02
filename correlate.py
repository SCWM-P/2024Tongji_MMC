import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


def readFile(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            results = f.read().strip().split('\n')
            results = [float(i) for i in results]
            return results
    else:
        raise FileNotFoundError(f'File not found! Path is {path}')


file_list = [
    'MSSIM', 'PSNR', 'PSNRc',
    'PSNRHA', 'PSNRHMA', 'PSNRHVS',
    'PSNRHVSM', 'WSNR', 'FSIMc',
    'SSIM', 'NSS'
]
standard_file_list = [
    'FSIM', 'FSIMc', 'MSSIM', 'SSIM',
    'PSNR', 'PSNRc', 'PSNRHA', 'PSNRHMA',
    'PSNRHVS', 'PSNRHVSM',
    'WSNR', 'VIFP', 'VSNR', 'NQM',
]
img_num = 25
distortion_types = 24
distortion_levels = 5
rootDir = os.getcwd()
savePath = os.path.join(rootDir, 'data', 'correlation')
readPath = os.path.join(rootDir, 'data', 'results')
standardPath = os.path.join(rootDir, 'tid2013', 'metrics_values')
mos = readFile(os.path.join(rootDir, 'tid2013', 'mos.txt'))

# 读取mos文件
mos = np.array(mos).reshape((img_num, distortion_types, distortion_levels))
# 读取results
results = {
    txt: np.array(readFile(os.path.join(readPath, f'{txt}.txt'))).reshape(
        (
            img_num,
            distortion_types,
            distortion_levels
        )
    )
    for txt in file_list
}
standard_results = {
    txt: np.array(readFile(os.path.join(standardPath, f'{txt}.txt'))).reshape(
        (
            img_num,
            distortion_types,
            distortion_levels
        )
    )
    for txt in standard_file_list
}

spearman = {
    txt: np.array([
        spearmanr(
            mos[:, i, :].flatten(),
            results[txt][:, i, :].flatten()
        )[0]
        for i in range(distortion_types)
    ])
    for txt in file_list
}
kendall = {
    txt: np.array([
        kendalltau(
            mos[:, i, :].flatten(),
            results[txt][:, i, :].flatten()
        )[0]
        for i in range(distortion_types)
    ])
    for txt in file_list
}
spearman_df = pd.DataFrame(spearman)
kendall_df = pd.DataFrame(kendall)

standard_spearman = {
    txt: np.array([
        spearmanr(
            mos[:, i, :].flatten(),
            standard_results[txt][:, i, :].flatten()
        )[0]
        for i in range(distortion_types)
    ])
    for txt in standard_file_list
}
standard_kendall = {
    txt: np.array([
        kendalltau(
            mos[:, i, :].flatten(),
            standard_results[txt][:, i, :].flatten()
        )[0]
        for i in range(distortion_types)
    ])
    for txt in standard_file_list
}
standard_spearman_df = pd.DataFrame(standard_spearman)
standard_kendall_df = pd.DataFrame(standard_kendall)

spearman_df.to_excel(os.path.join(savePath, 'spearman.xlsx'))
kendall_df.to_excel(os.path.join(savePath, 'kendall.xlsx'))
standard_spearman_df.to_excel(os.path.join(savePath, 'standard_spearman.xlsx'))
standard_kendall_df.to_excel(os.path.join(savePath, 'standard_kendall.xlsx'))