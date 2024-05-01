import os
import re
import sys
import cv2
import time
from scipy.stats import kendalltau, spearmanr
import numpy as np
import subprocess
import multiprocessing
from lib import img_process


def run_psnr(img_path, refer_img_path):
    python_exe = sys.executable
    run_arg = "-m"
    psnr_module = "psnr_hvsm"
    result = subprocess.run([python_exe, run_arg, psnr_module, img_path, refer_img_path],
                            capture_output=True, text=True)
    # if result.stderr:
    #     print("Error:{}".format(result.stderr))
    return result.stdout.strip()


def process_image(args):
    img_no, distortion_types, distortion_levels, reference_dir, distorted_dir, func_handler, files = args
    refer_img_path = os.path.join(reference_dir, f'I{img_no:02}.bmp')
    refer_img = cv2.imread(refer_img_path)
    outputs = np.zeros((distortion_types * distortion_levels, len(files)))

    for distortion_type in range(1, distortion_types + 1):
        for distortion_level in range(1, distortion_levels + 1):
            img_name = f'I{img_no:02}_{distortion_type:02}_{distortion_level}.bmp'
            img_path = os.path.join(distorted_dir, img_name)
            img = cv2.imread(img_path)
            psnr_output = run_psnr(img_path, refer_img_path)
            psnr_output = [float(i) for i in re.findall(r'=(\d+\.\d+)', psnr_output)]
            idx = (distortion_type - 1) * distortion_levels + (distortion_level - 1)
            outputs[idx, :] = [func(img, refer_img)[0] for func in func_handler] + psnr_output
            print(f'Processed {img_name} !')
    return outputs




def main():
    img_num = 25
    distortion_types = 24
    distortion_levels = 5
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'tid2013')
    reference_dir = os.path.join(data_dir, "reference_images")
    distorted_dir = os.path.join(data_dir, "distorted_images")
    files = [
        r'FSIMc.txt',
        r'MSSIM.txt',
        r'SSIM.txt',
        r'WSNR.txt',
        r'NSS.txt',
        r'PSNRc.txt',
        r'PSNRHVS.txt',
        r'PSNRHVSM.txt',
        r'PSNRHA.txt',
        r'PSNRHMA.txt',
        r'PSNR.txt'
    ]
    func_handler = [
        img_process.FSIMc,
        img_process.MSSIM,
        img_process.SSIM,
        img_process.WSNR,
        img_process.NSS,
        img_process.PSNRc
    ]
    outputs = np.random.rand(img_num * distortion_types * distortion_levels, len(files))

    # Set up multiprocessing pool with a controlled number of processes
    max_processes = 13
    # with multiprocessing.Pool(processes=max_processes) as pool:
    #     args = [
    #         (
    #             i, distortion_types,
    #             distortion_levels,
    #             reference_dir,
    #             distorted_dir,
    #             func_handler,
    #             files
    #         ) for i in range(1, img_num + 1)
    #     ]
    #     # Collect results from each process
    #     all_outputs = pool.map(process_image, args)
    #     # Aggregate results into one large array
    #     outputs = np.vstack(all_outputs)

    metrics_exe_dir = os.path.join(data_dir, "metrics_values")
    mos_file_path = os.path.join(metrics_exe_dir, 'mos.txt')
    with open(mos_file_path, 'r', encoding='utf-8') as f:
        mos = f.read()
    for i, filename in enumerate(files):
        file_path = os.path.join(current_dir, 'data', 'results', filename)
        np.savetxt(file_path, outputs[:, i], fmt='%.4f')
        kendall_output = kendalltau(mos, outputs[:, i])
        spearman_output = spearmanr(mos, outputs[:, i])
        # kendall_output = subprocess.run(
        #     [
        #         os.path.join(
        #             metrics_exe_dir,
        #             'kendall.exe'
        #         ),
        #         mos_file,
        #         file_path
        #     ],
        #     capture_output=True,
        #     text=True
        # ).stdout
        # spearman_output = subprocess.run(
        #     [
        #         os.path.join(
        #             metrics_exe_dir,
        #             'spearman.exe'
        #         ),
        #         mos_file,
        #         file_path
        #     ],
        #     capture_output=True,
        #     text=True
        # ).stdout
        print(f'The Kendall correlation of {filename[:-4]} algorithm is: \n{kendall_output}')
        print(f'The Spearman correlation of {filename[:-4]} algorithm is: \n{spearman_output}')

    print("All images processed.")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    start_time = time.time()
    main()
    timeString = f'Total time: {time.time() - start_time} seconds'
    print('='*len(timeString))
    print(timeString)
    print('='*len(timeString))
