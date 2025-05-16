import os
import statistics
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import eliminate_offset, pcd2depth, depth_overlay, report_avg_offset

data_dir = "datasets/test_5_14/in"

img_left_dir = os.path.join(data_dir, "cam2_img")
img_right_dir = os.path.join(data_dir, "cam3_img")

dist_path = os.path.join(data_dir, "avg_offset_dist_100.png")
stats_path = os.path.join(data_dir, "avg_offset_stats_100.txt")
avg_offset_path = os.path.join(data_dir, "avg_offset_arr_100.npy")
offset_log_path = os.path.join(data_dir, "offset_log.txt")

avg_offset_arr = []

with open(offset_log_path, "w") as f:
    for item in tqdm(os.listdir(img_left_dir)):
        item_name = os.path.splitext(item)[0]

        left_img_path = os.path.join(img_left_dir, f"{item_name}.png")
        right_img_path = os.path.join(img_right_dir, f"{item_name}.png")

        report_avg, avg_offset = report_avg_offset(left_img_path, right_img_path, item_name, block_h=3000, step=100)
        avg_offset_arr.append(avg_offset)
        f.write(f"{item}\t{report_avg}\n")

np.save(avg_offset_path, avg_offset_arr)

avg_offset_arr = np.load(avg_offset_path, allow_pickle=True).tolist()

print(sum(1 for x in avg_offset_arr if x > 100))

bins = np.arange(0, 101, 1)

plt.hist(avg_offset_arr, bins=bins, edgecolor='black')
plt.xlabel("avg vertical offset", fontsize=16)
plt.ylabel("num of frames", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(dist_path, dpi=300, bbox_inches='tight')


with open(stats_path, "w") as file:
    file.write(f"average offset: {statistics.mean(avg_offset_arr)} pixels\n")
    file.write(f"median offset: {statistics.median(avg_offset_arr)} pixels\n")
    file.write(f"min offset: {min(avg_offset_arr)} pixels\n")
    file.write(f"max offset: {max(avg_offset_arr)} pixels\n")
