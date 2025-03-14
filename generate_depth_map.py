import os
from utils import pcd2depth, depth_overlay, get_stats, pcd2depth1, get_stats1
import statistics
import numpy as np
import matplotlib.pyplot as plt
# import time
from tqdm import tqdm

# input directory
in_dir = "datasets/data/test_2_14/in"

cam2_dir = os.path.join(in_dir, "cam2_img")
cam3_dir = os.path.join(in_dir, "cam3_img")
pcd_dir = os.path.join(in_dir, "lidar")

# output directory
out_dir = "datasets/data/test_2_14/out"

out_depth_cam2 = os.path.join(out_dir, "cam2_depth")
out_depth_cam3 = os.path.join(out_dir, "cam3_depth")
out_depth_img_cam2 = os.path.join(out_dir, "cam2_depth_img")
out_depth_img_cam3 = os.path.join(out_dir, "cam3_depth_img")

os.makedirs(out_depth_cam2, exist_ok=True)
os.makedirs(out_depth_cam3, exist_ok=True)
os.makedirs(out_depth_img_cam2, exist_ok=True)
os.makedirs(out_depth_img_cam3, exist_ok=True)

# calibration matrices
cam2_in_mat = [[31470.1, 0, 2736, 0],
               [0, 30825, 1824, 0],
               [0, 0, 1.0, 0]]

cam2_ex_mat = [[0.999923, 0.00506971, 0.011448, -0.238761],
               [0.011292, -0.00537459, -0.999939, 0.000403294],
               [-0.00482625, 0.99984, -0.00586819, 0.0185998],
               [0, 0, 0, 1]]

# old extrinsic
# cam2_ex_mat = [[0.999933899272713, -0.003245217941172, 0.0111, -0.217316],
#                [0.010881133852003, -0.0108, -0.9999, -0.00038],
#                [0.003544976558669, 0.9998, -0.0112, 0.2076],
#                [0, 0, 0, 1]]

cam3_in_mat = [[31470.1, 0, 2736, 0],
               [0, 30825, 1824, 0],
               [0, 0, 1, 0]]

cam3_ex_mat = [[0.999799, -0.00445795, 0.0110814, 0.238761],
               [0.0108896, -0.00407825, -0.99995, 0.000403294],
               [0.00468311, 0.999844, -0.00444752, 0.0185998],
               [0, 0, 0, 1]]

# old extrinsic
# cam3_ex_mat = [[0.999799, -0.00445795, 0.0110814, 0.233303],
#                [0.0108896, -0.00407825, -0.99995, 0.0084903],
#                [0.00468311, 0.999844, -0.00444752, 0.207738],
#                [0, 0, 0, 1]]

matrices_cam2 = [cam2_ex_mat, cam2_in_mat]
matrices_cam3 = [cam3_ex_mat, cam3_in_mat]

# input images size
width = 5472
# height = 3500
height = 3648

count = 0

# read pcd file and generate gray scale depth map
print("Start processing point cloud")

for item in tqdm(os.listdir(pcd_dir)):
    pcd_path = os.path.join(pcd_dir, item)
    item_name = os.path.splitext(item)[0]

    depth_path_cam2 = os.path.join(out_depth_cam2, f"{item_name}_depth.png")
    depth_path_cam3 = os.path.join(out_depth_cam3, f"{item_name}_depth.png")

    pcd2depth1(pcd_path, width, height, cam2_in_mat, cam2_ex_mat, depth_path_cam2)
    pcd2depth1(pcd_path, width, height, cam3_in_mat, cam3_ex_mat, depth_path_cam3)

    # image_path_cam2 = os.path.join(cam2_dir, f"{item_name}.png")
    # image_path_cam3 = os.path.join(cam3_dir, f"{item_name}.png")
    #
    # depth_img_path_cam2 = os.path.join(out_depth_img_cam2, f"{item_name}_depth_img.png")
    # depth_img_path_cam3 = os.path.join(out_depth_img_cam3, f"{item_name}_depth_img.png")
    #
    # depth_overlay(depth_path_cam2, image_path_cam2, depth_img_path_cam2)
    # depth_overlay(depth_path_cam3, image_path_cam3, depth_img_path_cam3)

print("Finished processing point cloud")
exit()

# get stats from the dataset
depth_arr = []
count_arr = []
sample_count = 0

depth_arr_path = os.path.join(out_dir, "depth_arr.npy")
count_arr_path = os.path.join(out_dir, "count_arr.npy")

print("Start collecting stats")

# for item in tqdm(os.listdir(pcd_dir)):
#     sample_count += 1
#     pcd_path = os.path.join(pcd_dir, item)
#     cur_count, cur_depth_arr = get_stats1(pcd_path, width, height, cam2_in_mat, cam2_ex_mat)
#     count_arr.append(cur_count)
#     depth_arr += cur_depth_arr
#     cur_count, cur_depth_arr = get_stats1(pcd_path, width, height, cam3_in_mat, cam3_ex_mat)
#     count_arr.append(cur_count)
#     depth_arr += cur_depth_arr
#     print(f"Processing {sample_count}")
#
# np.save(depth_arr_path, depth_arr)
# np.save(count_arr_path, count_arr)
# print("Finished reading pcd, depth data saved")

depth_dist_path = os.path.join(out_dir, "depth_dist.png")
depth_stats_path = os.path.join(out_dir, "stats.txt")

depth_arr = np.load(depth_arr_path, allow_pickle=True).tolist()
count_arr = np.load(count_arr_path, allow_pickle=True).tolist()

bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
depth_count, bin_edges = np.histogram(depth_arr, bins=bins)
avg_depth_count = depth_count / len(count_arr)

plt.bar(bins[:-1], avg_depth_count, width=np.diff(bins), edgecolor='black', align='edge')
plt.xlabel("depth range")
plt.ylabel("num of points")
plt.savefig(depth_dist_path, dpi=300, bbox_inches='tight')

print("Done creating chart, start creating stats")

with open(depth_stats_path, "w") as file:
    file.write(f"size of dataset: {sample_count}\n")
    file.write(f"image size: {width}x{height}\n\n")
    file.write(f"average number of points: {statistics.mean(count_arr)}\n")
    file.write(f"median number of points: {statistics.median(count_arr)}\n")
    file.write(f"min number of points: {min(count_arr)}\n")
    file.write(f"max number of points: {max(count_arr)}\n\n")
    file.write(f"average depth of points: {statistics.mean(depth_arr)}\n")
    file.write(f"median depth of points: {statistics.median(depth_arr)}\n")
    file.write(f"min depth of points: {min(depth_arr)}\n")
    file.write(f"max depth of points: {max(depth_arr)}\n")

print("Finished generating stats")
