import os
from utils import pcd2depth, depth_overlay

cam2_dict = "datasets/data/cam2_img"
cam3_dict = "datasets/data/cam3_img"
pcd_dict = "datasets/data/lidar"
out_depth_cam2 = "datasets/out/cam2_depth"
out_depth_cam3 = "datasets/out/cam3_depth"
out_final_cam2 = "datasets/out/cam2_final"
out_final_cam3 = "datasets/out/cam3_final"

cam2_in_mat = [[31470.1, 0, 2736, 0],
               [0, 30825, 1824, 0],
               [0, 0, 1.0, 0]]

cam2_ex_mat = [[0.999933899272713, -0.003245217941172, 0.0111, -0.217316],
               [0.010881133852003, -0.0108, -0.9999, -0.00038],
               [0.003544976558669, 0.9998, -0.0112, 0.2076],
               [0, 0, 0, 1]]

cam3_in_mat = [[31470.1, 0, 2736, 0],
               [0, 30825, 1824, 0],
               [0, 0, 1, 0]]

cam3_ex_mat = [[0.999799, -0.00445795, 0.0110814, 0.233303],
               [0.0108896, -0.00407825, -0.99995, 0.0084903],
               [0.00468311, 0.999844, -0.00444752, 0.207738],
               [0, 0, 0, 1]]

width = 5472
height = 3648

count = 0

print("Starting processing point cloud")

for item in os.listdir(pcd_dict):
    pcd_path = os.path.join(pcd_dict, item)
    item_name, extension = os.path.splitext(item)

    print(f"Processing {item}")
    out_path_cam2 = os.path.join(out_depth_cam2, f"{item_name}_depth.png")
    out_path_cam3 = os.path.join(out_depth_cam3, f"{item_name}_depth.png")

    pcd2depth(pcd_path, width, height, cam2_in_mat, cam2_ex_mat, out_path_cam2)
    pcd2depth(pcd_path, width, height, cam3_in_mat, cam3_ex_mat, out_path_cam3)
    print(f"Done processing {item}")

    print(f"Overlaying depth points from {item}")
    image_path_cam2 = os.path.join(cam2_dict, f"{item_name}.png")
    image_path_cam3 = os.path.join(cam3_dict, f"{item_name}.png")
    out_path_final2 = os.path.join(out_final_cam2, f"{item_name}_final.png")
    out_path_final3 = os.path.join(out_final_cam3, f"{item_name}_final.png")

    depth_overlay(out_path_cam2, image_path_cam2, out_path_final2)
    depth_overlay(out_path_cam3, image_path_cam3, out_path_final3)
    print(f"Done overlaying depth points from {item}")

print("Finished")
