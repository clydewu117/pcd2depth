import os
import matplotlib.pyplot as plt
import numpy as np

from utils import pcd2disp
from tqdm import tqdm


img_dir = "datasets/data/test_2_14/in/cam3_img"
pcd_dir = "datasets/data/test_2_14/in/lidar"
out_dir = "datasets/data/test_2_14/disp"

# calibration matrices
cam2_in_mat = [[31470.1, 0, 2736, 0],
               [0, 30825, 1824, 0],
               [0, 0, 1.0, 0]]

cam2_ex_mat = [[0.999923, 0.00506971, 0.011448, -0.238761],
               [0.011292, -0.00537459, -0.999939, 0.000403294],
               [-0.00482625, 0.99984, -0.00586819, 0.0185998],
               [0, 0, 0, 1]]

# cam2_ex_mat = [[1, 0, 0, -0.238761],
#                [0, 0, -1, 0.000403294],
#                [0, 1, 0, 0.0185998],
#                [0, 0, 0, 1]]

cam3_in_mat = [[31470.1, 0, 2736, 0],
               [0, 30825, 1824, 0],
               [0, 0, 1, 0]]

cam3_ex_mat = [[0.999799, -0.00445795, 0.0110814, 0.238761],
               [0.0108896, -0.00407825, -0.99995, 0.000403294],
               [0.00468311, 0.999844, -0.00444752, 0.0185998],
               [0, 0, 0, 1]]

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
#     out_path = os.path.join(out_dir, f"{item_name}_disp.png")
#
#     neg, pos = pcd2disp(pcd_path, img_path, in_ex_left, in_ex_right, out_path)
#     neg_depth_arr += neg
#     pos_depth_arr += pos

# np.save("datasets/data/test_disp_map/neg_depth_arr.npy", np.array(neg_depth_arr))
# np.save("datasets/data/test_disp_map/pos_depth_arr.npy", np.array(pos_depth_arr))

neg_depth_arr = np.load("datasets/data/test_disp_map/neg_depth_arr.npy", allow_pickle=True).tolist()
pos_depth_arr = np.load("datasets/data/test_disp_map/pos_depth_arr.npy", allow_pickle=True).tolist()

min_neg, max_neg = min(neg_depth_arr), max(neg_depth_arr)
bins_neg = np.linspace(min_neg, max_neg, 50)

counts_neg, bins_neg = np.histogram(neg_depth_arr, bins=bins_neg)
normalized_counts_neg = counts_neg / 648

print(min(neg_depth_arr))
plt.bar(bins_neg[:-1], normalized_counts_neg, width=np.diff(bins_neg), edgecolor='black', align='edge')
plt.axvline(x=49, color='red', linestyle='-', label='beginning depth = 49')
plt.legend(fontsize=16)
plt.xlabel("depth", fontsize=16)
plt.ylabel("number of points", fontsize=16)
plt.title("depth where disparity is reversed", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

min_pos, max_pos = min(pos_depth_arr), max(pos_depth_arr)
bins_pos = np.linspace(min_pos, max_pos, 50)

counts_pos, bins_pos = np.histogram(pos_depth_arr, bins=bins_pos)
normalized_counts_pos = counts_pos / 648

plt.bar(bins_pos[:-1], normalized_counts_pos, width=np.diff(bins_pos), edgecolor='black', align='edge')
plt.xlabel("depth", fontsize=16)
plt.ylabel("number of points", fontsize=16)
plt.title("depth where disparity is normal", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
