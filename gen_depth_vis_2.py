import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import pcd2depth1, depth_overlay1
from tqdm import tqdm

data_dir = "datasets/test_6_27/in"
img_dir = os.path.join(data_dir, "cam3_img")
lidar1_dir = os.path.join(data_dir, "lidar_slam")
lidar2_dir = os.path.join(data_dir, "lidar_slam_2")

depth_dir_1 = "datasets/test_6_27/in/depth_1"
depth_dir_2 = "datasets/test_6_27/in/depth_2"

depth_vis_dir_1 = "datasets/test_6_27/in/depth_vis_1"
depth_vis_dir_2 = "datasets/test_6_27/in/depth_vis_2"

os.makedirs(depth_dir_1, exist_ok=True)
os.makedirs(depth_dir_2, exist_ok=True)
os.makedirs(depth_vis_dir_1, exist_ok=True)
os.makedirs(depth_vis_dir_2, exist_ok=True)

cam3_ex_mat = [
    [0.999639, 0.0180872, 0.0115616, 0.250893],
    [0.0104777, 0.0496413, -0.99873, -0.0072955],
    [-0.0184576, 0.998493, 0.0490144, 0.0234565],
    [0, 0, 0, 1],
]

cam2_ex_mat = [
    [0.999956, 0.00752824, 0.00586485, -0.25716],
    [0.00563727, 0.00588917, -0.999983, -0.00087487],
    [-0.00738311, 0.999828, 0.00540702, 0.0207079],
    [0, 0, 0, 1],
]

cam3_in_mat = [[30904.3, 0, 2446.95, 0], [0, 30142.5, 583.125, 0], [0, 0, 1, 0]]

cam2_in_mat = [[31204.7, 0, 2831.44, 0], [0, 30502.2, 1866.63, 0], [0, 0, 1.0, 0]]

# Initialize lists to collect statistics
stats_1 = {"counts": [], "means": [], "medians": [], "mins": [], "maxs": []}
stats_2 = {"counts": [], "means": [], "medians": [], "mins": [], "maxs": []}

# Initialize lists to collect all depth values for distribution plots
all_depths_1 = []
all_depths_2 = []

for scene in tqdm(sorted(os.listdir(img_dir))):
    name = os.path.splitext(scene)[0]
    img_path = os.path.join(img_dir, scene)
    lidar1_path = os.path.join(lidar1_dir, f"map_{name}.pcd")
    lidar2_path = os.path.join(lidar2_dir, f"map_{name}.1.pcd")

    depth_path_1 = os.path.join(depth_dir_1, f"{name}.png")
    depth_path_2 = os.path.join(depth_dir_2, f"{name}.png")

    depth_vis_path_1 = os.path.join(depth_vis_dir_1, f"{name}.png")
    depth_vis_path_2 = os.path.join(depth_vis_dir_2, f"{name}.png")

    # Use cam3 matrices for lidar_slam data and cam2 matrices for lidar_slam_2 data
    pcd2depth1(lidar1_path, cam3_in_mat, cam3_ex_mat, depth_path_1)
    pcd2depth1(lidar2_path, cam3_in_mat, cam3_ex_mat, depth_path_2)

    depth_overlay1(depth_path_1, img_path, depth_vis_path_1)
    depth_overlay1(depth_path_2, img_path, depth_vis_path_2)

    # Read and analyze depth maps
    # Analyze depth map 1 (from lidar_slam)
    depth_map_1 = cv2.imread(depth_path_1, cv2.IMREAD_UNCHANGED).astype(np.uint16)
    valid_depths_1 = (
        depth_map_1[depth_map_1 > 0] / 256.0
    )  # Convert to actual depth values

    if len(valid_depths_1) > 0:
        count_1 = len(valid_depths_1)
        mean_1 = np.mean(valid_depths_1)
        median_1 = np.median(valid_depths_1)
        min_1 = np.min(valid_depths_1)
        max_1 = np.max(valid_depths_1)

        stats_1["counts"].append(count_1)
        stats_1["means"].append(mean_1)
        stats_1["medians"].append(median_1)
        stats_1["mins"].append(min_1)
        stats_1["maxs"].append(max_1)

        # Collect all depth values for distribution plot
        all_depths_1.extend(valid_depths_1)

    # Analyze depth map 2 (from lidar_slam_2)
    depth_map_2 = cv2.imread(depth_path_2, cv2.IMREAD_UNCHANGED).astype(np.uint16)
    valid_depths_2 = (
        depth_map_2[depth_map_2 > 0] / 256.0
    )  # Convert to actual depth values

    if len(valid_depths_2) > 0:
        count_2 = len(valid_depths_2)
        mean_2 = np.mean(valid_depths_2)
        median_2 = np.median(valid_depths_2)
        min_2 = np.min(valid_depths_2)
        max_2 = np.max(valid_depths_2)

        stats_2["counts"].append(count_2)
        stats_2["means"].append(mean_2)
        stats_2["medians"].append(median_2)
        stats_2["mins"].append(min_2)
        stats_2["maxs"].append(max_2)

        # Collect all depth values for distribution plot
        all_depths_2.extend(valid_depths_2)

# Print overall summary statistics
print("\n" + "=" * 50)
print("OVERALL SUMMARY STATISTICS")
print("=" * 50)

if stats_1["counts"]:
    avg_count_1 = np.mean(stats_1["counts"])
    avg_mean_1 = np.mean(stats_1["means"])
    avg_median_1 = np.mean(stats_1["medians"])
    avg_min_1 = np.mean(stats_1["mins"])
    avg_max_1 = np.mean(stats_1["maxs"])

    print(
        f"\nDepth Map 1 (lidar_slam) - Average across {len(stats_1['counts'])} scenes:"
    )
    print(f"  Average Points per scene: {avg_count_1:.0f}")
    print(f"  Average Mean depth: {avg_mean_1:.2f}m")
    print(f"  Average Median depth: {avg_median_1:.2f}m")
    print(f"  Average Min depth: {avg_min_1:.2f}m")
    print(f"  Average Max depth: {avg_max_1:.2f}m")
else:
    print(f"\nDepth Map 1 (lidar_slam): No valid scenes processed")

if stats_2["counts"]:
    avg_count_2 = np.mean(stats_2["counts"])
    avg_mean_2 = np.mean(stats_2["means"])
    avg_median_2 = np.mean(stats_2["medians"])
    avg_min_2 = np.mean(stats_2["mins"])
    avg_max_2 = np.mean(stats_2["maxs"])

    print(
        f"\nDepth Map 2 (lidar_slam_2) - Average across {len(stats_2['counts'])} scenes:"
    )
    print(f"  Average Points per scene: {avg_count_2:.0f}")
    print(f"  Average Mean depth: {avg_mean_2:.2f}m")
    print(f"  Average Median depth: {avg_median_2:.2f}m")
    print(f"  Average Min depth: {avg_min_2:.2f}m")
    print(f"  Average Max depth: {avg_max_2:.2f}m")
else:
    print(f"\nDepth Map 2 (lidar_slam_2): No valid scenes processed")

# Generate depth distribution plots
print(f"\nGenerating depth distribution plots...")

# Generate separate plots for each dataset
# Plot 1: Lidar SLAM 1
if all_depths_1:
    plt.figure()
    plt.hist(all_depths_1, bins=50, alpha=0.7, color="blue", edgecolor="black")
    plt.xlabel("Depth (meters)")
    plt.ylabel("Number of Points")
    plt.title("Depth Distribution - Lidar SLAM 1")

    plt.savefig(
        "datasets/test_6_27/in/depth_distribution_lidar_slam_1.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

# Plot 2: Lidar SLAM 2
if all_depths_2:
    plt.figure()
    plt.hist(all_depths_2, bins=50, alpha=0.7, color="blue", edgecolor="black")
    plt.xlabel("Depth (meters)")
    plt.ylabel("Number of Points")
    plt.title("Depth Distribution - Lidar SLAM 2")

    plt.savefig(
        "datasets/test_6_27/in/depth_distribution_lidar_slam_2.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

print(f"Depth distribution plots saved to datasets/test_6_27/in/")
