import os
import cv2
from tqdm import tqdm

data_dir = "datasets/test_2_9/orig"
cam2_dir = os.path.join(data_dir, "cam2_img")
cam3_dir = os.path.join(data_dir, "cam3_img")

h_dir = "datasets/test_2_9/h"
cam2_dir_h = os.path.join(h_dir, "cam2_img")
cam3_dir_h = os.path.join(h_dir, "cam3_img")

q_dir = "datasets/test_2_9/q"
cam2_dir_q = os.path.join(q_dir, "cam2_img")
cam3_dir_q = os.path.join(q_dir, "cam3_img")

down_percent_h = 0.5
down_percent_q = 0.25


def downscale_images(input_dir, output_dir_h, output_dir_q):
    for item in tqdm(os.listdir(input_dir), desc=f"Processing {input_dir}"):
        img_path = os.path.join(input_dir, item)
        img = cv2.imread(img_path)

        new_size_h = (int(img.shape[1] * down_percent_h), int(img.shape[0] * down_percent_h))
        new_size_q = (int(img.shape[1] * down_percent_q), int(img.shape[0] * down_percent_q))

        resized_img_h = cv2.resize(img, new_size_h, interpolation=cv2.INTER_AREA)
        resized_img_q = cv2.resize(img, new_size_q, interpolation=cv2.INTER_AREA)

        cv2.imwrite(os.path.join(output_dir_h, item), resized_img_h)
        cv2.imwrite(os.path.join(output_dir_q, item), resized_img_q)


downscale_images(cam2_dir, cam2_dir_h, cam2_dir_q)
downscale_images(cam3_dir, cam3_dir_h, cam3_dir_q)


def downscale_depth_maps(input_dir, output_dir_h, output_dir_q):
    for item in tqdm(os.listdir(input_dir), desc=f"Processing {input_dir}"):
        depth_path = os.path.join(input_dir, item)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        new_size_h = (int(depth.shape[1] * down_percent_h), int(depth.shape[0] * down_percent_h))
        new_size_q = (int(depth.shape[1] * down_percent_q), int(depth.shape[0] * down_percent_q))

        resized_depth_h = cv2.resize(depth, new_size_h, interpolation=cv2.INTER_NEAREST)
        resized_depth_q = cv2.resize(depth, new_size_q, interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(os.path.join(output_dir_h, item), resized_depth_h, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite(os.path.join(output_dir_q, item), resized_depth_q, [cv2.IMWRITE_PNG_COMPRESSION, 0])


depth_dir = os.path.join(data_dir, "cam3_depth")
depth_dir_h = os.path.join(h_dir, "cam3_depth")
depth_dir_q = os.path.join(q_dir, "cam3_depth")

downscale_depth_maps(depth_dir, depth_dir_h, depth_dir_q)
