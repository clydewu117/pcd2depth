import os
import shutil

datasets_dir = "datasets"

src_dirs = []
for item in os.listdir(datasets_dir):
    if os.path.isdir(os.path.join(datasets_dir, item)) and item.startswith("data_") and item != "data_all":
        src_dirs.append(os.path.join(datasets_dir, item))

des_dir = "datasets/data_all"
des_cam2_dir = os.path.join(des_dir, "cam2_img")
des_cam3_dir = os.path.join(des_dir, "cam3_img")
des_lidar_dir = os.path.join(des_dir, "lidar")

log_file_path = os.path.join(des_dir, "correspondence_log.txt")

count = 0

print("Start copying files")

with open(log_file_path, "w") as log_file:
    for src_dir in src_dirs:

        src_cam2_dir = os.path.join(src_dir, "cam2_img")
        src_cam3_dir = os.path.join(src_dir, "cam3_img")
        src_lidar_dir = os.path.join(src_dir, "lidar")
        num_files = len([_ for _ in os.listdir(src_lidar_dir)])
        print(f"Copying {src_dir}, {num_files} samples in total")

        pre_count = count
        for item in sorted(os.listdir(src_lidar_dir)):
            src_name = os.path.splitext(item)[0]
            des_name = str(count)
            print(f"Copying {src_name} in src to {des_name} in des")

            src_cam2_img_path = os.path.join(src_cam2_dir, f"{src_name}.png")
            src_cam3_img_path = os.path.join(src_cam3_dir, f"{src_name}.png")
            src_lidar_path = os.path.join(src_lidar_dir, f"{src_name}.pcd")

            des_cam2_img_path = os.path.join(des_cam2_dir, f"{des_name}.png")
            des_cam3_img_path = os.path.join(des_cam3_dir, f"{des_name}.png")
            des_lidar_path = os.path.join(des_lidar_dir, f"{des_name}.pcd")

            shutil.copy2(src_cam2_img_path, des_cam2_img_path)
            shutil.copy2(src_cam3_img_path, des_cam3_img_path)
            shutil.copy2(src_lidar_path, des_lidar_path)

            log_file.write(f"{src_cam2_img_path} -> {des_cam2_img_path}\n")
            log_file.write(f"{src_cam3_img_path} -> {des_cam3_img_path}\n")
            log_file.write(f"{src_lidar_path} -> {des_lidar_path}\n\n")

            count += 1

        print(f"Finished copying {src_dir}")
        print(f"{src_dir} are listed from {pre_count} to {count - 1} in {des_dir}")

print("Finished")