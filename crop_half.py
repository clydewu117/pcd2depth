import os
import cv2
from tqdm import tqdm


def crop_lower_half(input_path, output_path=None):
    image = cv2.imread(input_path)
    height = image.shape[0]
    lower_half = image[height // 2:, :]

    if output_path:
        cv2.imwrite(output_path, lower_half)

    return lower_half


data_dir = "datasets/test_nov"
cam2_dir = os.path.join(data_dir, "cam2_img")
cam3_dir = os.path.join(data_dir, "cam3_img")

out_dir = "datasets/test_nov_lh"
cam2_out_dir = os.path.join(out_dir, "cam2_img")
cam3_out_dir = os.path.join(out_dir, "cam3_img")

os.makedirs(out_dir, exist_ok=True)
os.makedirs(cam2_out_dir, exist_ok=True)
os.makedirs(cam3_out_dir, exist_ok=True)

for item in tqdm(os.listdir(cam2_dir)):
    cam2_img_path = os.path.join(cam2_dir, item)
    cam3_img_path = os.path.join(cam3_dir, item)

    cam2_out_path = os.path.join(cam2_out_dir, item)
    cam3_out_path = os.path.join(cam3_out_dir, item)

    crop_lower_half(cam2_img_path, cam2_out_path)
    crop_lower_half(cam3_img_path, cam3_out_path)
