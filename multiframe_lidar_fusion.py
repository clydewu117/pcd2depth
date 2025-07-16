import os
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm


def multiframe_lidar_fusion(lidar_dir, out_path):
    """
    Fuse multiple LiDAR frames into a single point cloud and save it to a file.

    Parameters:
    - lidar_dir: Directory containing LiDAR point cloud files.
    - out_path: Output path for the fused point cloud.
    """

    # Check if lidar directory exists
    if not os.path.exists(lidar_dir):
        print(f"LiDAR directory does not exist: {lidar_dir}")
        return

    if not os.path.isdir(lidar_dir):
        print(f"Path is not a directory: {lidar_dir}")
        return

    # List all LiDAR files in the directory
    lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith(".pcd")])

    if not lidar_files:
        print(f"No LiDAR files found in {lidar_dir}")
        return

    # Initialize an empty point cloud
    fused_point_cloud = o3d.geometry.PointCloud()

    for lidar_file in lidar_files:
        file_path = os.path.join(lidar_dir, lidar_file)
        print(f"Loading {file_path}...")

        # Load the LiDAR point cloud
        pcd = o3d.io.read_point_cloud(file_path)

        # Append the points to the fused point cloud
        fused_point_cloud.points.extend(pcd.points)
        fused_point_cloud.colors.extend(pcd.colors)

    # Save the fused point cloud to the output path
    o3d.io.write_point_cloud(out_path, fused_point_cloud)
    print(f"Fused point cloud saved to {out_path}")


scene_dir = "datasets/test_6_20/scenes"

for scene in tqdm(sorted(os.listdir(scene_dir))):
    scene_path = os.path.join(scene_dir, scene)

    # Skip if not a directory (e.g., .DS_Store files)
    if not os.path.isdir(scene_path):
        continue

    lidar_dir = os.path.join(scene_path, "lidar")

    out_path = os.path.join(scene_path, f"{scene}_fused.pcd")

    multiframe_lidar_fusion(lidar_dir, out_path)
