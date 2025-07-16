import os
from tqdm import tqdm
import shutil

data_dir = "datasets/test_6_27"
img_dir = os.path.join(data_dir, "cam3_img")
lidar_dir = os.path.join(data_dir, "lidar")
out_dir = "datasets/test_6_27/in"
img_out_dir = os.path.join(out_dir, "cam3_img")
lidar_out_dir = os.path.join(out_dir, "lidar")
os.makedirs(img_out_dir, exist_ok=True)
os.makedirs(lidar_out_dir, exist_ok=True)

file_names = [0, 50, 70, 95, 131, 161, 193]
count = 1

for file_name in tqdm(file_names):

    img_path = os.path.join(img_dir, f"{file_name}.png")
    lidar_path = os.path.join(lidar_dir, f"{file_name}.pcd")
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} does not exist, skipping")
        continue
    out_img_path = os.path.join(img_out_dir, f"{count}.png")
    shutil.copy(img_path, out_img_path)

    if not os.path.exists(lidar_path):
        print(f"Warning: {lidar_path} does not exist, skipping")
        continue
    out_lidar_path = os.path.join(lidar_out_dir, f"{count}.pcd")
    shutil.copy(lidar_path, out_lidar_path)

    count += 1
