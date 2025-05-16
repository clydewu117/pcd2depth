import os
import numpy as np

from utils import pcd2disp
from tqdm import tqdm

dataset_dir = "datasets/test_5_14"

img_dir = os.path.join(dataset_dir, "in/cam3_img")
pcd_dir = os.path.join(dataset_dir, "in/lidar")
out_dir = os.path.join(dataset_dir, "disp")

os.makedirs(out_dir, exist_ok=True)

# May 14 calibration
cam3_ex_mat = [[
                        0.999639,
                        0.0180872,
                        0.0115616,
                        0.250893
                    ],
                    [
                        0.0104777,
                        0.0496413,
                        -0.99873,
                        -0.0072955
                    ],
                    [
                        -0.0184576,
                        0.998493,
                        0.0490144,
                        0.0234565
                    ],
                    [
                        0,
                        0,
                        0,
                        1
                    ]]

cam2_ex_mat = [[
                        0.999956,
                        0.00753269,
                        0.00585914,
                        -0.25716
                    ],
                    [
                        0.00563727,
                        0.00513034,
                        -0.999987,
                        -0.00087487
                    ],
                    [
                        -0.00738311,
                        0.999832,
                        0.00464831,
                        0.0207079
                    ],
                    [
                        0,
                        0,
                        0,
                        1
                    ]]

cam3_in_mat = [[30904.3, 0, 2446.95, 0],
               [0, 30142.5, 583.125, 0],
               [0, 0, 1.0, 0]]

cam2_in_mat = [[30895, 0, 2831.44, 0],
               [0, 29602.9, 1866.63, 0],
               [0, 0, 1.0, 0]]

# May 5th calibration
# cam3_in_mat = [[30904.3, 0, 2446.95, 0],
#                [0, 30142.5, 583.125, 0],
#                [0, 0, 1.0, 0]]
#
# cam3_ex_mat = [[0.999633, 0.0184489, 0.0115912, 0.239893],
#                [0.0104604, 0.0511609, -0.998654, -0.0972955],
#                [-0.0188366, 0.998411, 0.0505297, 0.0234565],
#                [0, 0, 0, 1]]
#
# cam2_in_mat = [[30895, 0, 2831.44, 0],
#                [0, 29602.9, 1866.63, 0],
#                [0, 0, 1.0, 0]]
#
# cam2_ex_mat = [[0.999928, 0.0105686, 0.00585463, -0.47716],
#                [0.0056185, 0.00502095, -0.999988, -0.00087487],
#                [-0.0104179, 0.999805, 0.00452189, 0.0207079],
#                [0, 0, 0, 1]]

# old calibration
# cam2_in_mat = [[31470.1, 0, 2736, 0],
#                [0, 30825, 1824, 0],
#                [0, 0, 1.0, 0]]
#
# cam2_ex_mat = [[0.999923, 0.00506971, 0.011448, -0.238761],
#                [0.011292, -0.00537459, -0.999939, 0.000403294],
#                [-0.00482625, 0.99984, -0.00586819, 0.0185998],
#                [0, 0, 0, 1]]

# cam2_ex_mat = [[1, 0, 0, -0.238761],
#                [0, 0, -1, 0.000403294],
#                [0, 1, 0, 0.0185998],
#                [0, 0, 0, 1]]

# cam3_in_mat = [[31470.1, 0, 2736, 0],
#                [0, 30825, 1824, 0],
#                [0, 0, 1, 0]]

# cam3_ex_mat = [[0.999799, -0.00445795, 0.0110814, 0.238761],
#                [0.0108896, -0.00407825, -0.99995, 0.000403294],
#                [0.00468311, 0.999844, -0.00444752, 0.0185998],
#                [0, 0, 0, 1]]

# cam3_ex_mat = [[1, 0, 0, 0.238761],
#                [0, 0, -1, 0.000403294],
#                [0, 1, 0, 0.0185998],
#                [0, 0, 0, 1]]

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

max_disp_arr = np.load(os.path.join(dataset_dir, "max_disp_arr.npy"), allow_pickle=True).tolist()
min_disp_arr = np.load(os.path.join(dataset_dir, "min_disp_arr.npy"), allow_pickle=True).tolist()

print(f"max disp: {max(max_disp_arr)}")
print(f"min disp: {min(min_disp_arr)}")

print(max_disp_arr)
print(min_disp_arr)
