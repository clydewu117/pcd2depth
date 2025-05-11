import os
from tqdm import tqdm
import shutil

dataset_dir = "datasets/test_5_5/in"

left_dir = os.path.join(dataset_dir, "cam3_img")
right_dir = os.path.join(dataset_dir, "cam2_img")

out_dir = os.path.join(dataset_dir, "comparison")
os.makedirs(out_dir, exist_ok=True)

for item in tqdm(os.listdir(left_dir)):
    name, ext = os.path.splitext(item)
    left_src_path = os.path.join(left_dir, item)
    right_src_path = os.path.join(right_dir, item)

    left_des_name = name + "_left.png"
    right_des_name = name + "_right.png"

    left_des_path = os.path.join(out_dir, left_des_name)
    right_des_path = os.path.join(out_dir, right_des_name)

    shutil.copy(left_src_path, left_des_path)
    shutil.copy(right_src_path, right_des_path)