import os
import numpy as np

from utils import pcd2depth1, depth_overlay1
from tqdm import tqdm

dataset_dir = "datasets/test_7_13_temp"

img_dir = os.path.join(dataset_dir, "in/cam3_img")
pcd_dir = os.path.join(dataset_dir, "in/lidar")
depth_dir = os.path.join(dataset_dir, "in/depth")
depth_vis_dir = os.path.join(dataset_dir, "in/depth_vis")
disp_dir = os.path.join(dataset_dir, "in/disp")
disp_vis_dir = os.path.join(dataset_dir, "in/disp_vis")

os.makedirs(depth_dir, exist_ok=True)
os.makedirs(depth_vis_dir, exist_ok=True)
os.makedirs(disp_vis_dir, exist_ok=True)

cam3_ex_mat = [
    [0.999727, 0.0169459, 8.10469e-05, 0.250893],
    [-0.000935544, 0.0496293, -0.998785, -0.0072955],
    [-0.0167537, 0.998513, 0.0492101, 0.0234565],
    [0, 0, 0, 1],
]

cam2_ex_mat = [
    [0.99996, 0.0070283, 0.00585774, -0.25716],
    [0.00564025, 0.00487455, -0.999988, -0.00087487],
    [-0.0068773, 0.999837, 0.0043954, 0.0207079],
    [0, 0, 0, 1],
]

cam3_in_mat = [[31370.2, 0, 2446.95, 0], [0, 30142.5, 583.125, 0], [0, 0, 1, 0]]
cam2_in_mat = [[31204.7, 0, 2831.44, 0], [0, 30502.2, 1866.63, 0], [0, 0, 1.0, 0]]

for item in tqdm(sorted(os.listdir(pcd_dir))):
    pcd_path = os.path.join(pcd_dir, item)
    item_name = os.path.splitext(item)[0]
    img_path = os.path.join(img_dir, f"{item_name}.png")
    depth_path = os.path.join(depth_dir, f"{item_name}.png")
    depth_vis_path = os.path.join(depth_vis_dir, f"{item_name}.png")
    disp_path = os.path.join(disp_dir, f"{item_name}.png")
    disp_vis_path = os.path.join(disp_vis_dir, f"{item_name}.png")

    # pcd2depth1(pcd_path, cam3_in_mat, cam3_ex_mat, depth_path)
    depth_overlay1(depth_path, img_path, depth_vis_path)
    depth_overlay1(disp_path, img_path, disp_vis_path)
