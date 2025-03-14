from utils import eliminate_offset
import os
import cv2

img_left_dir = "datasets/data/test_2_14/cross_test/cross_in/img1"
img_right_dir = "datasets/data/test_2_14/cross_test/cross_in/img2"

rec_left_dir = "datasets/data/test_2_14/cross_test/cross_uni/img1"
rec_right_dir = "datasets/data/test_2_14/cross_test/cross_uni/img2"

uni_rec_left_dir = "datasets/data/test_2_14/cross_test/cross_uni_rec/img1"
uni_rec_right_dir = "datasets/data/test_2_14/cross_test/cross_uni_rec/img2"

os.makedirs(rec_left_dir, exist_ok=True)
os.makedirs(rec_right_dir, exist_ok=True)
os.makedirs(uni_rec_left_dir, exist_ok=True)
os.makedirs(uni_rec_right_dir, exist_ok=True)

height_arr = []

for item in os.listdir(img_left_dir):
    item_name = os.path.splitext(item)[0]

    img_left_path = os.path.join(img_left_dir, f"{item_name}.png")
    img_right_path = os.path.join(img_right_dir, f"{item_name}.png")

    rec_left_path = os.path.join(rec_left_dir, f"{item_name}.png")
    rec_right_path = os.path.join(rec_right_dir, f"{item_name}.png")

    eliminate_offset(img_left_path, img_right_path, rec_left_path, rec_right_path)
    rec_img1 = cv2.imread(rec_left_path)
    height_arr.append(rec_img1.shape[0])

target_h = min(height_arr)

for item in os.listdir(rec_left_dir):
    item_name = os.path.splitext(item)[0]

    rec_left_path = os.path.join(rec_left_dir, f"{item_name}.png")
    rec_right_path = os.path.join(rec_right_dir, f"{item_name}.png")

    uni_rec_left_path = os.path.join(uni_rec_left_dir, f"{item_name}.png")
    uni_rec_right_path = os.path.join(uni_rec_right_dir, f"{item_name}.png")

    rec_img1 = cv2.imread(rec_left_path)
    rec_img2 = cv2.imread(rec_right_path)

    height = rec_img1.shape[0]
    h_diff = height - target_h

    uni_rec_img1 = rec_img1[h_diff:, :]
    uni_rec_img2 = rec_img2[h_diff:, :]

    cv2.imwrite(uni_rec_left_path, uni_rec_img1)
    cv2.imwrite(uni_rec_right_path, uni_rec_img2)
