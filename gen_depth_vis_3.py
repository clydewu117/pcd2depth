import os
import open3d as o3d
import numpy as np
from tqdm import tqdm
import shutil

scene_num = [13, 24, 36, 48, 61, 72, 84]
left_img_dir = "datasets/test_7_2/cam3_img"
right_img_dir = "datasets/test_7_2/cam2_img"

count = 0

for i in range(1, 10):
    if i == 1 or i == 4:
        continue
    scene_dir = f"datasets/test_7_2/in/scene_{count+1}"

    left_img_path = os.path.join(left_img_dir, f"{scene_num[count]}.png")
    right_img_path = os.path.join(right_img_dir, f"{scene_num[count]}.png")

    left_dest_path = os.path.join(scene_dir, "cam3_img", f"{scene_num[count]}.png")
    right_dest_path = os.path.join(scene_dir, "cam2_img", f"{scene_num[count]}.png")

    os.makedirs(os.path.dirname(left_dest_path), exist_ok=True)
    os.makedirs(os.path.dirname(right_dest_path), exist_ok=True)

    shutil.copy(left_img_path, left_dest_path)
    shutil.copy(right_img_path, right_dest_path)

    count += 1
