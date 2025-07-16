import os

from utils import pcd2depth1, depth_overlay1
from tqdm import tqdm

scene_dir = "datasets/test_6_20/scenes"

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

in_ex_left = cam3_in_mat, cam3_ex_mat
in_ex_right = cam2_in_mat, cam2_ex_mat

for scene in tqdm(sorted(os.listdir(scene_dir))):
    scene_path = os.path.join(scene_dir, scene)

    # Skip if not a directory (e.g., .DS_Store files)
    if not os.path.isdir(scene_path):
        continue

    lidar_dir = os.path.join(scene_path, "lidar")
    img_dir = os.path.join(scene_path, "cam3_img")
    depth_dir = os.path.join(scene_path, "depth")
    depth_vis_dir = os.path.join(scene_path, "depth_vis")
    depth_fused_vis_dir = os.path.join(scene_path, "depth_fused_vis")

    # Check if required directories exist
    if not os.path.exists(lidar_dir):
        print(f"LiDAR directory does not exist: {lidar_dir}")
        continue

    if not os.path.exists(img_dir):
        print(f"Image directory does not exist: {img_dir}")
        continue

    # Create output directories if they don't exist
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(depth_vis_dir, exist_ok=True)
    os.makedirs(depth_fused_vis_dir, exist_ok=True)

    # List all LiDAR files in the directory
    lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith(".pcd")])

    lidar_fused_path = os.path.join(scene_path, f"{scene}_fused.pcd")
    depth_fused_path = os.path.join(scene_path, f"{scene}_depth_fused.png")
    pcd2depth1(lidar_fused_path, cam3_in_mat, cam3_ex_mat, depth_fused_path)

    for lidar_file in tqdm(lidar_files, desc=f"Processing {scene}"):
        lidar_path = os.path.join(lidar_dir, lidar_file)
        img_path = os.path.join(img_dir, f"{os.path.splitext(lidar_file)[0]}.png")

        depth_path = os.path.join(depth_dir, f"{os.path.splitext(lidar_file)[0]}.png")
        depth_vis_path = os.path.join(
            depth_vis_dir, f"{os.path.splitext(lidar_file)[0]}.png"
        )
        depth_fused_vis_path = os.path.join(
            depth_fused_vis_dir, f"{os.path.splitext(lidar_file)[0]}.png"
        )

        # Convert LiDAR point cloud to depth image
        pcd2depth1(lidar_path, cam3_in_mat, cam3_ex_mat, depth_path)
        depth_overlay1(depth_path, img_path, depth_vis_path)
        depth_overlay1(depth_fused_path, img_path, depth_fused_vis_path)
