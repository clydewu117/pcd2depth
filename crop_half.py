import os
import cv2
from tqdm import tqdm


def crop_mid_half(input_path, output_path):
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    height = image.shape[0]
    mid_half = image[height//4:3*height//4, :]

    cv2.imwrite(output_path, mid_half)


data_dir = "datasets/test_5_5/in"
cam2_dir = os.path.join(data_dir, "cam2_img")
cam3_dir = os.path.join(data_dir, "cam3_img")
disp_dir = os.path.join(data_dir, "disp")

out_dir = "datasets/test_5_5/mid_half"
cam2_out_dir = os.path.join(out_dir, "cam2_img")
cam3_out_dir = os.path.join(out_dir, "cam3_img")
depth_out_dir = os.path.join(out_dir, "disp")

os.makedirs(out_dir, exist_ok=True)
os.makedirs(cam2_out_dir, exist_ok=True)
os.makedirs(cam3_out_dir, exist_ok=True)

for item in tqdm(os.listdir(cam2_dir)):

    cam2_img_path = os.path.join(cam2_dir, item)
    cam3_img_path = os.path.join(cam3_dir, item)
    disp_path = os.path.join(disp_dir, item)

    cam2_out_path = os.path.join(cam2_out_dir, item)
    cam3_out_path = os.path.join(cam3_out_dir, item)
    disp_out_path = os.path.join(depth_out_dir, item)

    crop_mid_half(cam2_img_path, cam2_out_path)
    crop_mid_half(cam3_img_path, cam3_out_path)
    crop_mid_half(disp_path, disp_out_path)
