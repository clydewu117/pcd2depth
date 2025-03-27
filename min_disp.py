import numpy as np

from utils import find_min_disp
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

pcd_dir = "datasets/data/test_2_14/in/lidar"

cam2_in_mat = [[31470.1, 0, 2736, 0],
               [0, 30825, 1824, 0],
               [0, 0, 1.0, 0]]

cam2_ex_mat = [[0.999923, 0.00506971, 0.011448, -0.238761],
               [0.011292, -0.00537459, -0.999939, 0.000403294],
               [-0.00482625, 0.99984, -0.00586819, 0.0185998],
               [0, 0, 0, 1]]

cam3_in_mat = [[31470.1, 0, 2736, 0],
               [0, 30825, 1824, 0],
               [0, 0, 1, 0]]

cam3_ex_mat = [[0.999799, -0.00445795, 0.0110814, 0.238761],
               [0.0108896, -0.00407825, -0.99995, 0.000403294],
               [0.00468311, 0.999844, -0.00444752, 0.0185998],
               [0, 0, 0, 1]]

height = 5472
width = 3648

disp_arr = []
w1_arr = []
w2_arr = []

for item in tqdm(os.listdir(pcd_dir)):
    pcd_path = os.path.join(pcd_dir, item)
    disp, w1, w2 = find_min_disp(pcd_path, cam2_ex_mat, cam2_in_mat, cam3_ex_mat, cam3_in_mat, height, width)
    disp_arr.append(disp)
    w1_arr.append(w1)
    w2_arr.append(w2)

for i in range(len(w1_arr)):
    print(w1_arr[i])
    print(w2_arr[i])
    print(disp_arr[i])
    print("\n")

np.save("datasets/data/min_disp/cam2_LR_dist.npy", np.array(w1_arr))
np.save("datasets/data/min_disp/cam3_LR_dist.npy", np.array(w2_arr))

plt.hist(disp_arr, bins=25, edgecolor='black')
plt.xlabel("min disparity")
plt.ylabel("number of frames")
plt.show()

plt.hist(w1_arr, bins=25, edgecolor='black')
plt.xlabel("depth of min disparity point")
plt.ylabel("number of frames")
plt.show()

plt.hist(w2_arr, bins=25, edgecolor='black')
plt.xlabel("depth of min disparity point")
plt.ylabel("number of frames")
plt.show()
