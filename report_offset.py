import os
import statistics
import matplotlib.pyplot as plt

from utils import eliminate_offset, pcd2depth, depth_overlay, report_offset


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

data_dir = "datasets/data/test_2_9/in"

img_left_dir = os.path.join(data_dir, "cam2_img")
img_right_dir = os.path.join(data_dir, "cam3_img")

log_path = os.path.join(data_dir, "offset_log.txt")
dist_path = os.path.join(data_dir, "offset_dist.png")
stats_path = os.path.join(data_dir, "offset_stats.txt")

offset_arr = []

with open(log_path, "w") as file:
    for item in os.listdir(img_left_dir):
        item_name = os.path.splitext(item)[0]

        left_img_path = os.path.join(img_left_dir, f"{item_name}.png")
        right_img_path = os.path.join(img_right_dir, f"{item_name}.png")

        report, offset = report_offset(left_img_path, right_img_path, item_name)
        file.write(f"{report}\n")
        offset_arr.append(offset)

plt.hist(offset_arr, bins=10, edgecolor='black')
plt.xlabel("offset")
plt.ylabel("num of samples")

plt.savefig(dist_path, dpi=300, bbox_inches='tight')

with open(stats_path, "w") as file:
    file.write(f"average offset: {statistics.mean(offset_arr)} pixels\n")
    file.write(f"median offset: {statistics.median(offset_arr)} pixels\n")
    file.write(f"min offset: {min(offset_arr)} pixels\n")
    file.write(f"max offset: {max(offset_arr)} pixels\n")
