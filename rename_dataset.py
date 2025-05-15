import os
from tqdm import tqdm

dataset_dir = "datasets/test_5_14"

cam2_dir = os.path.join(dataset_dir, "in/cam2_img")
cam3_dir = os.path.join(dataset_dir, "in/cam3_img")
pcd_dir = os.path.join(dataset_dir, "in/lidar")

for item in tqdm(os.listdir(pcd_dir)):
    name, _ = os.path.splitext(item)
    new_name = f"{int(name):06d}"

    cam2_path = os.path.join(cam2_dir, name + ".png")
    cam3_path = os.path.join(cam3_dir, name + ".png")
    pcd_path = os.path.join(pcd_dir, name + ".pcd")

    new_cam2_path = os.path.join(cam2_dir, new_name + ".png")
    new_cam3_path = os.path.join(cam3_dir, new_name + ".png")
    new_pcd_path = os.path.join(pcd_dir, new_name + ".pcd")

    os.rename(cam2_path, new_cam2_path)
    os.rename(cam3_path, new_cam3_path)
    os.rename(pcd_path, new_pcd_path)
