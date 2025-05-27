import os

from utils import pcd2disp_pn_lr
from tqdm import tqdm

dataset_dir = "datasets/test_5_5"

left_img_dir = os.path.join(dataset_dir, "in/cam3_img")
right_img_dir = os.path.join(dataset_dir, "in/cam2_img")
pcd_dir = os.path.join(dataset_dir, "in/lidar")
left_out_dir = os.path.join(dataset_dir, "pn_lr_disp/left_maps")
right_out_dir = os.path.join(dataset_dir, "pn_lr_disp/right_maps")

os.makedirs(left_out_dir, exist_ok=True)
os.makedirs(right_out_dir, exist_ok=True)

# May 12 calibration for test_5_5
cam3_ex_mat = [[
                        0.999613,
                        0.0194676,
                        0.0115788,
                        0.250893
                    ],
                    [
                        0.0104086,
                        0.05054,
                        -0.998686,
                        -0.0072955
                    ],
                    [
                        -0.0198467,
                        0.998423,
                        0.0498983,
                        0.0234565
                    ],
                    [
                        0,
                        0,
                        0,
                        1
                    ]]

cam2_ex_mat = [[
                        0.999941,
                        0.00929654,
                        0.00586799,
                        -0.25716
                    ],
                    [
                        0.00562645,
                        0.00627855,
                        -0.999981,
                        -0.00087487
                    ],
                    [
                        -0.00915342,
                        0.999811,
                        0.00578637,
                        0.0207079
                    ],
                    [
                        0,
                        0,
                        0,
                        1
                    ]]

cam3_in_mat = [[30904.3, 0, 2446.95, 0],
               [0, 30142.5, 583.125, 0],
               [0, 0, 1.0, 0]]

cam2_in_mat = [[30895, 0, 2831.44, 0],
               [0, 29602.9, 1866.63, 0],
               [0, 0, 1.0, 0]]


in_ex_left = cam3_in_mat, cam3_ex_mat
in_ex_right = cam2_in_mat, cam2_ex_mat

neg_depth_arr = []
pos_depth_arr = []

for item in tqdm(sorted(os.listdir(pcd_dir))):
    pcd_path = os.path.join(pcd_dir, item)
    item_name = os.path.splitext(item)[0]
    left_img_path = os.path.join(left_img_dir, f"{item_name}.png")
    right_img_path = os.path.join(right_img_dir, f"{item_name}.png")
    left_out_path = os.path.join(left_out_dir, f"{item_name}_pn_disp.png")
    right_out_path = os.path.join(right_out_dir, f"{item_name}_pn_disp.png")

    pcd2disp_pn_lr(pcd_path, left_img_path, right_img_path, in_ex_left, in_ex_right, left_out_path, right_out_path)
