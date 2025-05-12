import os
import matplotlib.pyplot as plt
import numpy as np

from utils import pcd2disp_pn
from tqdm import tqdm

dataset_dir = "datasets/test_5_5"

img_dir = "datasets/test_5_5/in/cam3_img"
pcd_dir = "datasets/test_5_5/in/lidar"
out_dir = "datasets/test_5_5/pn_disp1/maps"

os.makedirs(out_dir, exist_ok=True)

# May 12 calibration
cam3_ex_mat = [[
                        0.999613,
                        0.0194676,
                        0.0115788,
                        0.250893
                    ],
                    [
                        0.0104086,
                        0.05054,
                        -0.998686,
                        -0.0072955
                    ],
                    [
                        -0.0198467,
                        0.998423,
                        0.0498983,
                        0.0234565
                    ],
                    [
                        0,
                        0,
                        0,
                        1
                    ]]

cam2_ex_mat = [[
                        0.999941,
                        0.00929654,
                        0.00586799,
                        -0.25716
                    ],
                    [
                        0.00562645,
                        0.00627855,
                        -0.999981,
                        -0.00087487
                    ],
                    [
                        -0.00915342,
                        0.999811,
                        0.00578637,
                        0.0207079
                    ],
                    [
                        0,
                        0,
                        0,
                        1
                    ]]

cam2_in_mat = [[30895, 0, 2831.44, 0],
               [0, 29602.9, 1866.63, 0],
               [0, 0, 1.0, 0]]

cam3_in_mat = [[30904.3, 0, 2446.95, 0],
               [0, 30142.5, 583.125, 0],
               [0, 0, 1.0, 0]]

# May 5th calibration
# cam2_in_mat = [[30895, 0, 2831.44, 0],
#                [0, 29602.9, 1866.63, 0],
#                [0, 0, 1.0, 0]]
#
# cam2_ex_mat = [[0.999928, 0.0105686, 0.00585463, -0.47716],
#                [0.0056185, 0.00502095, -0.999988, -0.00087487],
#                [-0.0104179, 0.999805, 0.00452189, 0.0207079],
#                [0, 0, 0, 1]]
#
# cam3_in_mat = [[30904.3, 0, 2446.95, 0],
#                [0, 30142.5, 583.125, 0],
#                [0, 0, 1.0, 0]]
#
# cam3_ex_mat = [[0.999633, 0.0184489, 0.0115912, 0.239893],
#                [0.0104604, 0.0511609, -0.998654, -0.0972955],
#                [-0.0188366, 0.998411, 0.0505297, 0.0234565],
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

neg_depth_arr = []
pos_depth_arr = []

# for item in tqdm(os.listdir(pcd_dir)):
#     pcd_path = os.path.join(pcd_dir, item)
#     item_name = os.path.splitext(item)[0]
#     img_path = os.path.join(img_dir, f"{item_name}.png")
#     out_path = os.path.join(out_dir, f"{item_name}_pn_disp.png")
#
#     neg, pos = pcd2disp_pn(pcd_path, img_path, in_ex_left, in_ex_right, out_path)
#     neg_depth_arr += neg
#     pos_depth_arr += pos
#
# np.save("datasets/test_5_5/pn_disp/neg_depth_arr.npy", np.array(neg_depth_arr))
# np.save("datasets/test_5_5/pn_disp/pos_depth_arr.npy", np.array(pos_depth_arr))

neg_depth_arr = np.load("datasets/test_5_5/pn_disp/neg_depth_arr.npy", allow_pickle=True).tolist()
pos_depth_arr = np.load("datasets/test_5_5/pn_disp/pos_depth_arr.npy", allow_pickle=True).tolist()

file_count = len(os.listdir(img_dir))

min_neg, max_neg = min(neg_depth_arr), max(neg_depth_arr)
bins_neg = np.arange(0, 501, 10)

counts_neg, bins_neg = np.histogram(neg_depth_arr, bins=bins_neg)
normalized_counts_neg = counts_neg / file_count

print(min(neg_depth_arr))
plt.figure()
plt.bar(bins_neg[:-1], normalized_counts_neg, width=np.diff(bins_neg), edgecolor='black', align='edge')
plt.xlabel("depth", fontsize=16)
plt.ylabel("number of points", fontsize=16)
plt.title("depth where disparity is reversed", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig("datasets/test_5_5/pn_disp/neg_disp.png", dpi=300)

min_pos, max_pos = min(pos_depth_arr), max(pos_depth_arr)
bins_pos = np.arange(0, 501, 10)

counts_pos, bins_pos = np.histogram(pos_depth_arr, bins=bins_pos)
normalized_counts_pos = counts_pos / file_count

plt.figure()
plt.bar(bins_pos[:-1], normalized_counts_pos, width=np.diff(bins_pos), edgecolor='black', align='edge')
plt.xlabel("depth", fontsize=16)
plt.ylabel("number of points", fontsize=16)
plt.title("depth where disparity is normal", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig("datasets/test_5_5/pn_disp/pos_disp.png", dpi=300)
