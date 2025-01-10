import os
from utils import pcd2depth, depth_overlay, get_stats
import statistics
import numpy as np
import matplotlib.pyplot as plt
# import time


cam2_dir = "datasets/data_all/cam2_img"
cam3_dir = "datasets/data_all/cam3_img"
pcd_dir = "datasets/data_raw/lidar"
out_depth_cam2 = "datasets/depth_img/cam2_depth"
out_depth_cam3 = "datasets/depth_img/cam3_depth"
out_depth_img_cam2 = "datasets/depth_img/cam2_depth_img"
out_depth_img_cam3 = "datasets/depth_img/cam3_depth_img"

cam2_in_mat = [[31470.1, 0, 2736, 0],
               [0, 30825, 1824, 0],
               [0, 0, 1.0, 0]]

cam2_ex_mat = [[0.999933899272713, -0.003245217941172, 0.0111, -0.217316],
               [0.010881133852003, -0.0108, -0.9999, -0.00038],
               [0.003544976558669, 0.9998, -0.0112, 0.2076],
               [0, 0, 0, 1]]

cam3_in_mat = [[31470.1, 0, 2736, 0],
               [0, 30825, 1824, 0],
               [0, 0, 1, 0]]

cam3_ex_mat = [[0.999799, -0.00445795, 0.0110814, 0.233303],
               [0.0108896, -0.00407825, -0.99995, 0.0084903],
               [0.00468311, 0.999844, -0.00444752, 0.207738],
               [0, 0, 0, 1]]

matrices_cam2 = [cam2_ex_mat, cam2_in_mat]
matrices_cam3 = [cam3_ex_mat, cam3_in_mat]

width = 5472
height = 3648

count = 0

# read pcd file and generate gray scale depth map
# print("Start processing point cloud")
#
# for item in os.listdir(pcd_dir):
#     pcd_path = os.path.join(pcd_dir, item)
#     item_name = os.path.splitext(item)[0]
#
#     print(f"Processing {item}")
#     depth_path_cam2 = os.path.join(out_depth_cam2, f"{item_name}_depth.png")
#     depth_path_cam3 = os.path.join(out_depth_cam3, f"{item_name}_depth.png")
#
#     pcd2depth(pcd_path, width, height, cam2_in_mat, cam2_ex_mat, depth_path_cam2)
#     pcd2depth(pcd_path, width, height, cam3_in_mat, cam3_ex_mat, depth_path_cam3)
#     print(f"Done processing {item}")
#
#     print(f"Overlaying depth points from {item}")
#     image_path_cam2 = os.path.join(cam2_dir, f"{item_name}.png")
#     image_path_cam3 = os.path.join(cam3_dir, f"{item_name}.png")
#     depth_img_path_cam2 = os.path.join(out_depth_img_cam2, f"{item_name}_depth_img.png")
#     depth_img_path_cam3 = os.path.join(out_depth_img_cam3, f"{item_name}_depth_img.png")
#
#     depth_overlay(depth_path_cam2, image_path_cam2, depth_img_path_cam2)
#     depth_overlay(depth_path_cam3, image_path_cam3, depth_img_path_cam3)
#     print(f"Done overlaying depth points from {item}")
#
# print("Finished processing point cloud")

# get stats from the dataset
count_arr = []
depth_arr = []
sample_count = 0

print("Starting collecting stats")

for item in os.listdir(pcd_dir):
    sample_count += 1
    pcd_path = os.path.join(pcd_dir, item)
    cur_count, cur_depth_arr = get_stats(pcd_path, width, height, cam2_in_mat, cam2_ex_mat)
    count_arr.append(cur_count)
    depth_arr += cur_depth_arr
    cur_count, cur_depth_arr = get_stats(pcd_path, width, height, cam3_in_mat, cam3_ex_mat)
    count_arr.append(cur_count)
    depth_arr += cur_depth_arr

bins = [0, 100, 200, 300, 400, 500]
depth_count, bin_edges = np.histogram(depth_arr, bins=bins)
avg_depth_count = depth_count / sample_count

plt.bar(bins[:-1], avg_depth_count, width=np.diff(bins), edgecolor='black', align='edge')
plt.xlabel("depth range")
plt.ylabel("num of points")
plt.savefig('depth_dist.png', dpi=300, bbox_inches='tight')

with open("stats.txt", "w") as file:
    file.write(f"size of dataset: {sample_count}\n")
    file.write(f"image size: {width}x{height}\n")
    file.write(f"average number of points: {statistics.mean(count_arr)}\n")
    file.write(f"median number of points: {statistics.median(count_arr)}\n")
    file.write(f"average depth of points: {statistics.mean(depth_arr)}\n")
    file.write(f"median depth of points: {statistics.median(depth_arr)}\n")
    file.write(f"min depth of points: {min(depth_arr)}\n")
    file.write(f"max depth of points: {max(depth_arr)}\n")

print("Finished collecting stats")

# report noise
# noises = []
#
# start_time = time.time()
#
# for item in os.listdir(pcd_dir):
#     pcd_path = os.path.join(pcd_dir, item)
#     item_name, extension = os.path.splitext(item)
#     image_path_cam2 = os.path.join(cam2_dir, f"{item_name}.png")
#     image_path_cam3 = os.path.join(cam3_dir, f"{item_name}.png")
#     noises += report_noise(pcd_path, image_path_cam2, image_path_cam3, width, height, matrices_cam2, matrices_cam3)
#
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Elapsed time: {elapsed_time} seconds")
# print(f"points: {len(noises)}")
# print(f"points rgb diff > 10000: {len([x for x in noises if x > 10000])}")
#
# plt.hist(noises, bins=100, edgecolor='black', alpha=0.7)
# plt.show()
