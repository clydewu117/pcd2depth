import os
import shutil
import tempfile


def fix_scene_numbering(scene_dir):
    """Fix the numbering in a single scene directory"""
    cam2_dir = os.path.join(scene_dir, "cam2_img")
    cam3_dir = os.path.join(scene_dir, "cam3_img")
    lidar_dir = os.path.join(scene_dir, "lidar")

    # Check if directories exist
    dirs_to_process = []
    for dir_path, dir_name in [
        (cam2_dir, "cam2_img"),
        (cam3_dir, "cam3_img"),
        (lidar_dir, "lidar"),
    ]:
        if os.path.exists(dir_path):
            dirs_to_process.append((dir_path, dir_name))

    if not dirs_to_process:
        print(f"  No subdirectories found in {scene_dir}")
        return

    # Get all existing files and sort them
    all_files = {}
    for dir_path, dir_name in dirs_to_process:
        files = [f for f in os.listdir(dir_path) if f.endswith((".png", ".pcd"))]
        files.sort()
        all_files[dir_name] = files
        print(f"  Found {len(files)} files in {dir_name}")

    # Find the maximum number of files to determine how many sets we have
    max_files = max(len(files) for files in all_files.values()) if all_files else 0

    if max_files == 0:
        print(f"  No files found in {scene_dir}")
        return

    # Create temporary directories for renaming
    temp_dirs = {}
    for dir_path, dir_name in dirs_to_process:
        temp_dir = tempfile.mkdtemp()
        temp_dirs[dir_name] = temp_dir

    try:
        # Move files to temporary directories with new sequential names
        files_renamed = 0
        for dir_path, dir_name in dirs_to_process:
            files = all_files[dir_name]
            temp_dir = temp_dirs[dir_name]

            for idx, filename in enumerate(files):
                old_path = os.path.join(dir_path, filename)
                # Determine new filename based on file extension
                if filename.endswith(".png"):
                    new_filename = f"{idx:06d}.png"
                elif filename.endswith(".pcd"):
                    new_filename = f"{idx:06d}.pcd"
                else:
                    continue

                temp_path = os.path.join(temp_dir, new_filename)
                shutil.move(old_path, temp_path)
                files_renamed += 1

        # Move files back with correct sequential naming
        for dir_path, dir_name in dirs_to_process:
            temp_dir = temp_dirs[dir_name]
            temp_files = os.listdir(temp_dir)
            temp_files.sort()

            for filename in temp_files:
                temp_path = os.path.join(temp_dir, filename)
                new_path = os.path.join(dir_path, filename)
                shutil.move(temp_path, new_path)

        print(f"  Successfully renumbered {files_renamed} files")

    finally:
        # Clean up temporary directories
        for temp_dir in temp_dirs.values():
            try:
                os.rmdir(temp_dir)
            except:
                pass


def main():
    scenes_dir = "datasets/test_6_20/scenes"

    if not os.path.exists(scenes_dir):
        print(f"Error: Scenes directory '{scenes_dir}' does not exist!")
        return

    # Get all scene directories
    scene_folders = [
        d
        for d in os.listdir(scenes_dir)
        if os.path.isdir(os.path.join(scenes_dir, d)) and d.startswith("scene_")
    ]
    scene_folders.sort()

    if not scene_folders:
        print("No scene folders found!")
        return

    print(f"Found {len(scene_folders)} scene folders")
    print("Starting numbering fix...")
    print()

    for scene_folder in scene_folders:
        scene_path = os.path.join(scenes_dir, scene_folder)
        print(f"Processing {scene_folder}...")
        fix_scene_numbering(scene_path)
        print()

    print("Numbering fix completed!")
    print("All files in each scene are now numbered sequentially starting from 000000")


if __name__ == "__main__":
    main()
