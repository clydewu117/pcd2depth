from utils import eliminate_offset
import os
import cv2
from tqdm import tqdm

dataset_dir = "datasets/test_7_13_temp"

cam2_img_dir = os.path.join(dataset_dir, "in/cam2_img")
cam3_img_dir = os.path.join(dataset_dir, "in/cam3_img")
disp_dir = os.path.join(dataset_dir, "in/disp")
depth_dir = os.path.join(dataset_dir, "in/depth")

uni_rec_cam2_dir = os.path.join(dataset_dir, "in_uni_rec/cam2_img")
uni_rec_cam3_dir = os.path.join(dataset_dir, "in_uni_rec/cam3_img")
uni_rec_disp_dir = os.path.join(dataset_dir, "in_uni_rec/disp")
uni_rec_depth_dir = os.path.join(dataset_dir, "in_uni_rec/depth")

os.makedirs(uni_rec_cam2_dir, exist_ok=True)
os.makedirs(uni_rec_cam3_dir, exist_ok=True)
os.makedirs(uni_rec_disp_dir, exist_ok=True)
os.makedirs(uni_rec_depth_dir, exist_ok=True)

height_arr = []
cam2_img_dict = {}
cam3_img_dict = {}
disp_dict = {}
depth_dict = {}

for item in tqdm(os.listdir(cam2_img_dir)):

    cam2_img_path = os.path.join(cam2_img_dir, item)
    cam3_img_path = os.path.join(cam3_img_dir, item)
    disp_path = os.path.join(disp_dir, item)
    depth_path = os.path.join(depth_dir, item)

    cam2_img = cv2.imread(cam2_img_path, cv2.IMREAD_UNCHANGED)
    cam3_img = cv2.imread(cam3_img_path, cv2.IMREAD_UNCHANGED)
    disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    offset = eliminate_offset(cam2_img, cam3_img)

    remain_h = cam2_img.shape[0] - offset
    rec_cam2_img = cam2_img[:remain_h, :]
    rec_cam3_img = cam3_img[offset:, :]
    disp = disp[offset:, :]
    depth = depth[offset:, :]

    cam2_img_dict[item] = rec_cam2_img
    cam3_img_dict[item] = rec_cam3_img
    disp_dict[item] = disp
    depth_dict[item] = depth

    height_arr.append(remain_h)

min_h = min(height_arr)

for item in tqdm(cam2_img_dict.keys()):
    cam2_img = cam2_img_dict[item]
    cam3_img = cam3_img_dict[item]
    disp = disp_dict[item]
    depth = depth_dict[item]

    height = cam2_img.shape[0]
    h_diff = height - min_h

    uni_rec_cam2_img = cam2_img[h_diff:, :]
    uni_rec_cam3_img = cam3_img[h_diff:, :]
    uni_rec_disp = disp[h_diff:, :]
    uni_rec_depth = depth[h_diff:, :]

    cv2.imwrite(os.path.join(uni_rec_cam2_dir, item), uni_rec_cam2_img)
    cv2.imwrite(os.path.join(uni_rec_cam3_dir, item), uni_rec_cam3_img)
    cv2.imwrite(os.path.join(uni_rec_disp_dir, item), uni_rec_disp)
    cv2.imwrite(os.path.join(uni_rec_depth_dir, item), uni_rec_depth)
