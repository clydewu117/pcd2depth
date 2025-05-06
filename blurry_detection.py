import numpy as np
import cv2
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt


def fft_process(image, draw=False):
    img_gray = np.mean(image, axis=2) / 255.0
    fft_result = np.fft.fft2(img_gray)
    fft_shifted = np.fft.fftshift(np.abs(fft_result))

    if draw:
        fft_shifted = np.log(1 + fft_shifted)

    return fft_shifted / np.max(fft_shifted)


def sharpness_score(m):
    w, h = m.shape
    center = (w / 2, h / 2)
    max_d = np.log(1 + center[0] * center[1])

    y, x = np.indices((w, h))
    score_mat = np.log(1 + np.abs(y - center[0]) * np.abs(x - center[1])) / max_d

    return np.sum(score_mat * m)


data_dir = "datasets/test_5_5/May_5_RoadTest"
cam2_dir = os.path.join(data_dir, "cam2_img")
cam3_dir = os.path.join(data_dir, "cam3_img")

cam2_sc_dict = {}
cam3_sc_dict = {}
cam2_dict_path = os.path.join(data_dir, "cam2_sharpness.json")
cam3_dict_path = os.path.join(data_dir, "cam3_sharpness.json")

for item in tqdm(os.listdir(cam2_dir)):
    img_path = os.path.join(cam2_dir, item)
    img = cv2.imread(img_path)[:, :, ::-1]
    score = sharpness_score(fft_process(img))
    cam2_sc_dict[item] = score


for item in tqdm(os.listdir(cam3_dir)):
    img_path = os.path.join(cam3_dir, item)
    img = cv2.imread(img_path)[:, :, ::-1]
    score = sharpness_score(fft_process(img))
    cam3_sc_dict[item] = score


with open(cam2_dict_path, "w") as f:
    json.dump(cam2_sc_dict, f, indent=4)

with open(cam3_dict_path, "w") as f:
    json.dump(cam3_sc_dict, f, indent=4)

bins = np.arange(0, 251, 1)
cam2_sharpness_path = os.path.join(data_dir, "cam2_sharpness.png")
cam3_sharpness_path = os.path.join(data_dir, "cam3_sharpness.png")

cam2_sharpness = list(cam2_sc_dict.values())
cam3_sharpness = list(cam3_sc_dict.values())

plt.hist(cam2_sharpness, bins=bins, edgecolor='black')
plt.xlabel("sharpness score", fontsize=16)
plt.ylabel("num of frames", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(cam2_sharpness_path, dpi=300, bbox_inches='tight')

plt.hist(cam3_sharpness, bins=bins, edgecolor='black')
plt.xlabel("sharpness score", fontsize=16)
plt.ylabel("num of frames", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(cam3_sharpness_path, dpi=300, bbox_inches='tight')

print(f"cam2 number of sharpness > 250 = {sum(1 for _ in cam2_sharpness if _ > 250)}")
print(f"cam3 number of sharpness > 250 = {sum(1 for _ in cam3_sharpness if _ > 250)}")
