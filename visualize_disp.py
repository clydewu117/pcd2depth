import os
import cv2
import numpy as np
from tqdm import tqdm

disp_dir = "datasets/test_5_5/in/disp"
img_dir = "datasets/test_5_5/in/cam3_img"
out_dir = "datasets/test_5_5/in/disp_vis"

os.makedirs(out_dir, exist_ok=True)


def visualize_disp(disp_path, img_path, out_path):
    disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    valid_mask = disp > 0
    v, u = np.where(valid_mask)
    disp = disp[v, u] / 65535 * 1000
    disp_norm = (disp / disp.max()).astype(np.uint8)

    disp_gray = np.zeros_like(disp, dtype=np.uint8)
    disp_gray[v, u] = disp_norm

    color_map = cv2.applyColorMap(disp_gray, cv2.COLORMAP_JET)
    color_map_rgb = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

    for i in range(len(v)):
        color = tuple(map(int, color_map_rgb[v[i], u[i]]))
        center = (u[i], v[i])
        cv2.circle(img, center, radius=5, color=color, thickness=-1)

    image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, image_bgr)


for item in tqdm(os.listdir(disp_dir)):
    disp_path = os.path.join(disp_dir, item)
    img_path = os.path.join(img_dir, item)
    out_path = os.path.join(out_dir, item)

    visualize_disp(disp_path, img_path, out_dir)
