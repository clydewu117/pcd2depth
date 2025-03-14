import os
import statistics
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import report_avg_offset

data_dir = "datasets/KITTI/training"

img_left_dir = os.path.join(data_dir, "image_3")
img_right_dir = os.path.join(data_dir, "image_2")

dist_path = os.path.join(data_dir, "avg_offset_dist_100.png")
stats_path = os.path.join(data_dir, "avg_offset_stats_100.txt")
avg_offset_path = os.path.join(data_dir, "avg_offset_arr_100.npy")

avg_offset_arr = []

for item in tqdm(os.listdir(img_left_dir)):

    left_img_path = os.path.join(img_left_dir, item)
    right_img_path = os.path.join(img_right_dir, item)

    report_avg, avg_offset = report_avg_offset(left_img_path, right_img_path, item, block_h=200, step=25)
    avg_offset_arr.append(avg_offset)

np.save(avg_offset_path, avg_offset_arr)

avg_offset_arr = np.load(avg_offset_path, allow_pickle=True).tolist()

plt.hist(avg_offset_arr, bins=10, edgecolor='black')
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