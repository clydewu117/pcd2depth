import os
import shutil

groups = {
    "scene_1": [0, 20],
    "scene_2": [21, 55],
    "scene_3": [56, 76],
    "scene_4": [77, 96],
    "scene_5": [97, 118],
    "scene_6": [119, 140],
    "scene_7": [141, 162],
    "scene_8": [163, 184],
    "scene_9": [194, 206],
    "scene_10": [207, 228],
    "scene_11": [229, 250],
    "scene_12": [251, 271],
    "scene_13": [272, 292],
    "scene_14": [293, 311],
    "scene_15": [315, 336],
    "scene_16": [337, 356],
    "scene_17": [361, 380],
    "scene_18": [381, 402],
    "scene_19": [403, 424],
    "scene_20": [425, 446],
    "scene_21": [448, 468],
    "scene_22": [469, 490],
    "scene_23": [491, 511],
    "scene_24": [512, 533],
    "scene_25": [534, 555],
    "scene_26": [556, 577],
    "scene_27": [579, 599],
    "scene_28": [601, 621],
    "scene_29": [622, 664],
    "scene_30": [666, 686],
    "scene_31": [687, 701],
    "scene_32": [713, 733],
    "scene_33": [734, 755],
    "scene_34": [756, 765],
    "scene_35": [767, 787],
    "scene_36": [788, 809],
    "scene_37": [810, 834],
    "scene_38": [836, 856],
    "scene_39": [859, 878],
    "scene_40": [882, 900],
    "scene_41": [901, 921],
    "scene_42": [922, 943],
    "scene_43": [944, 964],
    "scene_44": [965, 986],
    "scene_45": [987, 1007],
}

data_dir = "datasets/test_6_20/in"

# Validate input directories exist
if not os.path.exists(data_dir):
    print(f"Error: Data directory '{data_dir}' does not exist!")
    exit(1)

cam3_dir = os.path.join(data_dir, "cam3_img")
cam2_dir = os.path.join(data_dir, "cam2_img")
lidar_dir = os.path.join(data_dir, "lidar")

# Check if source directories exist
missing_dirs = []
for dir_path, dir_name in [
    (cam3_dir, "cam3_img"),
    (cam2_dir, "cam2_img"),
    (lidar_dir, "lidar"),
]:
    if not os.path.exists(dir_path):
        missing_dirs.append(dir_name)

if missing_dirs:
    print(f"Error: Missing source directories: {', '.join(missing_dirs)}")
    exit(1)

out_dir = "datasets/test_6_20/scenes"
print(f"Starting file grouping from '{data_dir}' to '{out_dir}'")
print(f"Processing {len(groups)} scenes...")
print()

for scene_range in groups:
    scene, (start, end) = scene_range, groups[scene_range]

    cur_scene_dir = os.path.join(out_dir, scene)
    os.makedirs(cur_scene_dir, exist_ok=True)

    cur_scene_cam3_dir = os.path.join(cur_scene_dir, "cam3_img")
    cur_scene_cam2_dir = os.path.join(cur_scene_dir, "cam2_img")
    cur_scene_lidar_dir = os.path.join(cur_scene_dir, "lidar")
    os.makedirs(cur_scene_cam3_dir, exist_ok=True)
    os.makedirs(cur_scene_cam2_dir, exist_ok=True)
    os.makedirs(cur_scene_lidar_dir, exist_ok=True)

    count = 0

    files_moved = 0
    files_missing = 0

    for i in range(start, end + 1):
        scene_cam3_path = os.path.join(cur_scene_cam3_dir, f"{count:06d}.png")
        scene_cam2_path = os.path.join(cur_scene_cam2_dir, f"{count:06d}.png")
        scene_lidar_path = os.path.join(cur_scene_lidar_dir, f"{count:06d}.pcd")

        # Source file paths
        src_cam3_path = os.path.join(cam3_dir, f"{i:06d}.png")
        src_cam2_path = os.path.join(cam2_dir, f"{i:06d}.png")
        src_lidar_path = os.path.join(lidar_dir, f"{i:06d}.pcd")

        # Move files if they exist
        if os.path.exists(src_cam3_path):
            shutil.move(src_cam3_path, scene_cam3_path)
            print(f"Moved {src_cam3_path} -> {scene_cam3_path}")
            files_moved += 1
        else:
            print(f"Warning: {src_cam3_path} not found")
            files_missing += 1

        if os.path.exists(src_cam2_path):
            shutil.move(src_cam2_path, scene_cam2_path)
            print(f"Moved {src_cam2_path} -> {scene_cam2_path}")
            files_moved += 1
        else:
            print(f"Warning: {src_cam2_path} not found")
            files_missing += 1

        if os.path.exists(src_lidar_path):
            shutil.move(src_lidar_path, scene_lidar_path)
            print(f"Moved {src_lidar_path} -> {scene_lidar_path}")
            files_moved += 1
        else:
            print(f"Warning: {src_lidar_path} not found")
            files_missing += 1

        count += 1

    print(
        f"Completed {scene}: {files_moved} files moved, {files_missing} files missing"
    )

print("File grouping completed!")
print(f"All scenes processed successfully.")
