import os

import cv2

from utils import report_misalignment
import matplotlib.pyplot as plt

# depth = cv2.imread("datasets/out/cam2_depth/0_depth.png")
# image = cv2.imread("datasets/data/cam2_img/0.png")
#
# edges_depth = cv2.Canny(depth, 100, 200)
# edges_img_120 = cv2.Canny(image, 100, 120)
# edges_img_150 = cv2.Canny(image, 100, 150)
# edges_img_180 = cv2.Canny(image, 100, 180)
# edges_img_200 = cv2.Canny(image, 100, 200)
#
# cv2.imwrite("datasets/data/mis/edge_depth.png", edges_depth)
# cv2.imwrite("datasets/data/mis/edge_img120.png", edges_img_120)
# cv2.imwrite("datasets/data/mis/edge_img150.png", edges_img_150)
# cv2.imwrite("datasets/data/mis/edge_img180.png", edges_img_180)
# cv2.imwrite("datasets/data/mis/edge_img200.png", edges_img_200)

image_dir = "datasets/data/test_2_9/in/cam2_img"
depth_dir = "datasets/data/test_2_9/out/cam2_depth"

for item in os.listdir(image_dir):
    image_path = os.path.join(image_dir, item)
    item_name = os.path.splitext(item)[0]
    depth_path = os.path.join(depth_dir, f"{item_name}_depth.png")

    print(f"Processing {item}")

    report_misalignment(image_path, depth_path)
