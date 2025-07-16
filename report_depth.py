import os
import cv2
import numpy as np
from tqdm import tqdm

data_dir = "datasets/test_6_20/scenes"

with open(os.path.join(data_dir, "scene_depth_stats.txt"), "w") as f:
    for scene in tqdm(sorted(os.listdir(data_dir))):
        scene_path = os.path.join(data_dir, scene)

        # Skip if not a directory (e.g., .DS_Store files)
        if not os.path.isdir(scene_path):
            continue

        depth_dir = os.path.join(scene_path, "depth")

        # Check if depth directory exists
        if not os.path.exists(depth_dir):
            print(f"Depth directory does not exist: {depth_dir}")
            continue

        total_depth_points = 0
        depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".png")])

        print(f"\nProcessing {scene}:")

        depth_fused_path = os.path.join(scene_path, f"{scene}_depth_fused.png")
        depth_fused = cv2.imread(depth_fused_path, cv2.IMREAD_UNCHANGED)

        depth_fused_points = np.count_nonzero(depth_fused)

        for depth_file in depth_files:
            depth_path = os.path.join(depth_dir, depth_file)

            # Load depth map (use IMREAD_UNCHANGED to preserve depth values)
            depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            if depth_map is None:
                print(f"Could not load depth map: {depth_path}")
                continue

            # Count non-zero depth points (valid depth measurements)
            non_zero_points = np.count_nonzero(depth_map)
            total_depth_points += non_zero_points

            print(f"  {depth_file}: {non_zero_points} depth points")

        print(f"Total depth points in {scene}: {total_depth_points}")
        if depth_files:
            avg_points_per_frame = total_depth_points / len(depth_files)
            f.write(f"{scene} num frames: {len(depth_files)}\n")
            f.write(f"{scene} average num depth points: {avg_points_per_frame:.2f}\n")
            f.write(f"{scene} fused num depth points: {depth_fused_points}\n\n")
