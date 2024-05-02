import os
import cv2
from lib import img_process, MSSIM_process, NSS_process, PSNR_process, WSNR_process

current_dir = os.getcwd()
data_dir = os.path.join(current_dir, 'tid2013')
reference_dir = os.path.join(data_dir, "reference_images")
distorted_dir = os.path.join(data_dir, "distorted_images")
file_path = os.path.join(os.getcwd(), 'pictures')
img_refer = cv2.imread(os.path.join(reference_dir, 'I22.bmp'))
img_distorted = cv2.imread(os.path.join(distorted_dir, 'I22_23_5.bmp'))

if __name__ == '__main__':
    MSSIM_process.MSSIM_visual(img_distorted, img_refer, file_path)
    PSNR_process.PSNR_visual(img_distorted, img_refer, file_path)
    WSNR_process.WSNR_visual(img_distorted, img_refer, file_path)
    NSS_process.NSS_visual(img_distorted, file_path)
