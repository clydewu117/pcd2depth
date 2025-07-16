import os
import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

data_dir = "datasets/test_6_20/scenes"

# Define depth ranges (meter by meter)
max_depth = 500  # Analyze up to 500 meters
depth_ranges = [(i, i + 1) for i in range(max_depth)]
depth_ranges.append((max_depth, np.inf))  # Beyond max_depth

range_labels = [f"{i}-{i+1}m" for i in range(max_depth)]
range_labels.append(f"{max_depth}m+")

# Store points count for each range across all scenes
range_counts = {label: [] for label in range_labels}
total_points_per_scene = []
scene_names = []
all_max_depths = []  # Track all maximum depths
all_depth_values = []  # Track all individual depth values for overall statistics

for scene in tqdm(sorted(os.listdir(data_dir))):
    scene_path = os.path.join(data_dir, scene)

    # Skip if not a directory
    if not os.path.isdir(scene_path):
        continue

    fused_depth_path = os.path.join(scene_path, f"{scene}_depth_fused.png")

    # Skip if depth file doesn't exist
    if not os.path.exists(fused_depth_path):
        continue

    depth = cv2.imread(fused_depth_path, cv2.IMREAD_UNCHANGED)

    if depth is None:
        continue

    depth = depth.astype(np.float32) / 256

    # Get valid depth points (non-zero values)
    valid_depth_mask = depth > 0
    valid_depths = depth[valid_depth_mask]

    scene_names.append(scene)
    total_points_per_scene.append(len(valid_depths))

    # Track maximum depth found and collect all depth values
    if len(valid_depths) > 0:
        scene_max_depth = np.max(valid_depths)
        all_max_depths.append(scene_max_depth)
        print(f"{scene}: max depth = {scene_max_depth:.2f}m")

        # Collect all depth values for overall statistics
        all_depth_values.extend(valid_depths.tolist())

    # Count points in each depth range
    for i, (min_depth, max_depth_range) in enumerate(depth_ranges):
        range_mask = (valid_depths >= min_depth) & (valid_depths < max_depth_range)
        points_in_range = np.sum(range_mask)
        range_counts[range_labels[i]].append(points_in_range)

# Calculate average points per depth range
avg_points_per_range = {}
for label in range_labels:
    if range_counts[label]:
        avg_points_per_range[label] = np.mean(range_counts[label])
    else:
        avg_points_per_range[label] = 0

# Create visualization following gen_depth_map.py format
depth_bins = list(range(max_depth + 1))  # 0, 1, 2, ..., max_depth
avg_counts = [avg_points_per_range[f"{i}-{i+1}m"] for i in range(max_depth)]

plt.figure()
plt.bar(
    depth_bins[:-1],
    avg_counts,
    width=np.diff(depth_bins),
    edgecolor="black",
    align="edge",
)
plt.xlabel("depth range", fontsize=16)
plt.ylabel("num of points", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig("depth_points_by_meter_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

# Print statistics
print("=== Depth Points Analysis by Range ===")
print(f"Total scenes analyzed: {len(scene_names)}")
print(f"Average total points per scene: {np.mean(total_points_per_scene):.2f}")

# Print maximum depth statistics
if all_max_depths:
    print(f"\n=== Maximum Depth Statistics ===")
    print(f"Overall maximum depth found: {np.max(all_max_depths):.2f}m")
    print(f"Average maximum depth per scene: {np.mean(all_max_depths):.2f}m")
    print(f"Minimum maximum depth: {np.min(all_max_depths):.2f}m")

# Print overall depth statistics across all points
if all_depth_values:
    print(f"\n=== Overall Depth Statistics Across All Points ===")
    print(f"Total depth points analyzed: {len(all_depth_values):,}")
    print(f"Maximum depth: {np.max(all_depth_values):.2f}m")
    print(f"Minimum depth: {np.min(all_depth_values):.2f}m")
    print(f"Average depth: {np.mean(all_depth_values):.2f}m")
    print(f"Median depth: {np.median(all_depth_values):.2f}m")
    print(f"Standard deviation: {np.std(all_depth_values):.2f}m")

    # Calculate percentage of points above 150m
    points_above_150m = np.sum(np.array(all_depth_values) > 150)
    percentage_above_150m = (points_above_150m / len(all_depth_values)) * 100
    print(f"\nPoints above 150m: {points_above_150m:,} ({percentage_above_150m:.2f}%)")
    print(
        f"Points at or below 150m: {len(all_depth_values) - points_above_150m:,} ({100 - percentage_above_150m:.2f}%)"
    )
print("\nAverage points per depth range:")
for label, avg_count in avg_points_per_range.items():
    percentage = (
        (avg_count / np.mean(total_points_per_scene)) * 100
        if np.mean(total_points_per_scene) > 0
        else 0
    )
    print(f"  {label}: {avg_count:.2f} points ({percentage:.1f}%)")

# Additional statistics
print(
    f"\nDepth range with most points: {max(avg_points_per_range, key=avg_points_per_range.get)}"
)
print(
    f"Depth range with least points: {min(avg_points_per_range, key=avg_points_per_range.get)}"
)
