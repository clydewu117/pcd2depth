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
depth_dir = os.path.join(data_dir, "cam3_depth")

out_dir = "datasets/test_2_9_lh"
cam2_out_dir = os.path.join(out_dir, "cam2_img")
cam3_out_dir = os.path.join(out_dir, "cam3_img")
depth_out_dir = os.path.join(out_dir, "cam3_depth")

os.makedirs(out_dir, exist_ok=True)
os.makedirs(cam2_out_dir, exist_ok=True)
os.makedirs(cam3_out_dir, exist_ok=True)

for item in tqdm(os.listdir(cam2_dir)):
    item_name = os.path.splitext(item)[0]

    cam2_img_path = os.path.join(cam2_dir, item)
    cam3_img_path = os.path.join(cam3_dir, item)
    depth_path = os.path.join(depth_dir, f"{item_name}_depth.png")

    cam2_out_path = os.path.join(cam2_out_dir, item)
    cam3_out_path = os.path.join(cam3_out_dir, item)
    depth_out_path = os.path.join(depth_out_dir, f"{item_name}_depth.png")

    crop_lower_half(cam2_img_path, cam2_out_path)
    crop_lower_half(cam3_img_path, cam3_out_path)
    crop_lower_half(depth_path, depth_out_path)
