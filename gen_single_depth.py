import os
import cv2
import numpy as np
from utils import pcd2depth1, depth_overlay1

img_path = "datasets/data/test_2_14/in/cam2_img/111.png"
depth_path = "datasets/data/test_2_14/out_new_ex/cam2_depth/111_depth.png"
out_path = "datasets/data/test_2_14/out_new_ex/cam2_depth_img/111_depth_img.png"


width = 5472
height = 3648

# calibration matrices
cam2_in_mat = [[31470.1, 0, 2736, 0],
               [0, 30825, 1824, 0],
               [0, 0, 1.0, 0]]

# cam2_ex_mat = [[0.999923, 0.00506971, 0.011448, -0.238761],
#                [0.011292, -0.00537459, -0.999939, 0.000403294],
#                [-0.00482625, 0.99984, -0.00586819, 0.0185998],
#                [0, 0, 0, 1]]

# new ex
cam2_ex_mat = [[1, 0, 0, -0.217316],
               [0, 0, -1, -0.00038],
               [0, 1, 0, 0.2076],
               [0, 0, 0, 1]]

depth_overlay1(depth_path, img_path, out_path)
