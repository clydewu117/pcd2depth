import os
import statistics
import matplotlib.pyplot as plt

from utils import eliminate_offset, pcd2depth, depth_overlay, report_offset

img_path_1 = 'datasets/data/cam2_img/0.png'
img_path_2 = 'datasets/data/cam3_img/0.png'
pcd_path = 'datasets/data/lidar/0.pcd'
save_path = 'datasets/data/test'
depth_path = 'datasets/data/test_depth/0_depth.pcd'
img_left_path = 'datasets/data/test/cropped_image_left.png'
depth_img_path = 'datasets/data/test_depth_img/0_depth_img.png'

width = 5472
height = 3461
in_mat = [[31470.1, 0, 2736, 0],
               [0, 30825, 1824, 0],
               [0, 0, 1.0, 0]]

ex_mat = [[0.999933899272713, -0.003245217941172, 0.0111, -0.217316],
               [0.010881133852003, -0.0108, -0.9999, -0.00038],
               [0.003544976558669, 0.9998, -0.0112, 0.2076],
               [0, 0, 0, 1]]

# eliminate_offset(img_path_1, img_path_2, save_path)
#
# pcd2depth(pcd_path, width, height, in_mat, ex_mat, depth_path)
#
# depth_overlay(depth_path, img_left_path, depth_img_path)

img_left_dict = 'datasets/data_raw/cam2_img'
img_right_dict = 'datasets/data_raw/cam3_img'

offset_arr = []

for item in os.listdir(img_left_dict):
    item_name = os.path.splitext(item)[0]

    left_img_path = os.path.join(img_left_dict, f"{item_name}.png")
    right_img_path = os.path.join(img_right_dict, f"{item_name}.png")

    offset = report_offset(left_img_path, right_img_path)
    offset_arr.append(offset)

plt.hist(offset_arr, bins=10, edgecolor='black')
plt.xlabel("offset")
plt.ylabel("num of samples")

plt.savefig('offset_dist.png', dpi=300, bbox_inches='tight')

with open("offset_stats.txt", "w") as file:
    file.write(f"average offset: {statistics.mean(offset_arr)}\n")
    file.write(f"median offset: {statistics.median(offset_arr)}\n")
    file.write(f"min offset: {min(offset_arr)}\n")
    file.write(f"max offset: {max(offset_arr)}\n")
