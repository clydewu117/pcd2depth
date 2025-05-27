import os
import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

disp_dir = "datasets/test_5_5/in/disp"


def readDispOSU(filename):
    disp = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(np.float32)
    valid = disp > 0.0
    disp = disp / 65535 * 1000
    return disp, valid


focal_length = 30900
baseline = 0.51

conversion_factor = focal_length * baseline
depth_arr = []

for item in tqdm(os.listdir(disp_dir)):
    disp_path = os.path.join(disp_dir, item)
    disp_img, valid = readDispOSU(disp_path)

    depth_img = disp_img.copy()
    depth_img[~valid] = 0  # set invalid pixels to 0
    # convert disparity to depth
    depth_img = conversion_factor / depth_img / 70
    # set invalid pixels to 0
    depth_img[~valid] = 0
    depth_arr.extend(depth_img[valid].flatten())

bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
depth_count, bin_edges = np.histogram(depth_arr, bins=bins)
avg_depth_count = depth_count / 164

plt.bar(bins[:-1], avg_depth_count, width=np.diff(bins), edgecolor='black', align='edge')
plt.xlabel("depth range")
plt.ylabel("num of points")
plt.savefig("datasets/test_5_5/in/depth_dist.png", dpi=300, bbox_inches='tight')
