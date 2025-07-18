import os
import numpy as np

from utils import pcd2disp
from tqdm import tqdm

dataset_dir = "datasets/test_7_13_temp"

img_dir = os.path.join(dataset_dir, "in/cam3_img")
pcd_dir = os.path.join(dataset_dir, "in/lidar")
out_dir = os.path.join(dataset_dir, "in/disp")

os.makedirs(out_dir, exist_ok=True)

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

in_ex_left = cam3_in_mat, cam3_ex_mat
in_ex_right = cam2_in_mat, cam2_ex_mat

max_disp_arr = []
min_disp_arr = []

for item in tqdm(sorted(os.listdir(pcd_dir))):
    pcd_path = os.path.join(pcd_dir, item)
    item_name = os.path.splitext(item)[0]
    img_path = os.path.join(img_dir, f"{item_name}.png")
    out_path = os.path.join(out_dir, f"{item_name}.png")

    max_disp, min_disp = pcd2disp(pcd_path, in_ex_left, in_ex_right, out_path)

    max_disp_arr.append(max_disp)
    min_disp_arr.append(min_disp)

np.save(os.path.join(dataset_dir, "max_disp_arr.npy"), np.array(max_disp_arr))
np.save(os.path.join(dataset_dir, "min_disp_arr.npy"), np.array(min_disp_arr))

max_disp_arr = np.load(
    os.path.join(dataset_dir, "max_disp_arr.npy"), allow_pickle=True
).tolist()
min_disp_arr = np.load(
    os.path.join(dataset_dir, "min_disp_arr.npy"), allow_pickle=True
).tolist()

print(f"max disp: {max(max_disp_arr)}")
print(f"min disp: {min(min_disp_arr)}")

print(max_disp_arr)
print(min_disp_arr)
