import os.path

import open3d as o3d
import numpy as np
import cv2
import png
import statistics
import tqdm
# import math
# from collections import Counter
# from PIL import Image


def pcd2depth(pcd_path, width, height, in_mat, ex_mat, out_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)  # (N, 3)

    in_mat = np.array(in_mat)
    ex_mat = np.array(ex_mat)

    # Convert to homogeneous coordinates (N, 4)
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))

    points_cam = ex_mat @ points_h.T
    points_cam = points_cam.T

    # Camera to pixel transformation (3, 4) x (4, N) → (3, N)
    points_px = in_mat @ points_cam.T
    points_px = points_px.T

    # Normalize to get pixel coordinates
    w = points_px[:, 2]
    u = np.round(points_px[:, 0] / w).astype(int)
    v = np.round(points_px[:, 1] / w).astype(int)

    # Filter valid pixel coordinates
    valid_mask = (0 <= u) & (u < width) & (0 <= v) & (v < height) & (w < 500)
    u, v, w = u[valid_mask], v[valid_mask], w[valid_mask]

    # Normalize depth values for better visualization
    max_depth = np.max(w)
    # depth_norm = 500
    w_uint16 = (w / max_depth * 65535).astype(np.uint16)

    depth_map = np.full((height, width), 65535, dtype=np.uint16)

    for i in range(len(u)):
        cv2.circle(depth_map, (u[i], v[i]), 5, int(w_uint16[i]), thickness=-1)
        depth_map[v[i], u[i]] = np.minimum(w_uint16[i], depth_map[v[i], u[i]])

    depth_map[depth_map == 65535] = 0
    max_depth = np.max(depth_map)
    # max_depth = 500
    non_zero_mask = depth_map > 0
    depth_map[non_zero_mask] = max_depth - depth_map[non_zero_mask]

    with open(out_path, 'wb') as f:
        writer = png.Writer(width=depth_map.shape[1],
                            height=depth_map.shape[0],
                            bitdepth=16,
                            greyscale=True)
        writer.write(f, depth_map.tolist())


def pcd2depth1(pcd_path, width, height, in_mat, ex_mat, out_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)  # (N, 3)

    in_mat = np.array(in_mat)
    ex_mat = np.array(ex_mat)

    # Convert to homogeneous coordinates (N, 4)
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))

    points_cam = ex_mat @ points_h.T
    points_cam = points_cam.T

    # Camera to pixel transformation (3, 4) x (4, N) → (3, N)
    points_px = in_mat @ points_cam.T
    points_px = points_px.T

    # Normalize to get pixel coordinates
    w = points_px[:, 2]
    u = np.round(points_px[:, 0] / w).astype(int)
    v = np.round(points_px[:, 1] / w).astype(int)

    # Filter valid pixel coordinates
    valid_mask = (0 <= u) & (u < width) & (0 <= v) & (v < height) & (w < 500)
    u, v, w = u[valid_mask], v[valid_mask], w[valid_mask]

    # Normalize depth values for better visualization
    # max_depth = np.max(w)
    depth_norm = 500
    w_uint16 = (w / depth_norm * 65535).astype(np.uint16)

    depth_map = np.full((height, width), 65535, dtype=np.uint16)

    for i in range(len(u)):
        depth_map[v[i], u[i]] = np.minimum(w_uint16[i], depth_map[v[i], u[i]])

    depth_map[depth_map == 65535] = 0
    # max_depth = np.max(depth_map)
    max_depth = 500
    non_zero_mask = depth_map > 0
    depth_map[non_zero_mask] = max_depth - depth_map[non_zero_mask]

    with open(out_path, 'wb') as f:
        writer = png.Writer(width=depth_map.shape[1],
                            height=depth_map.shape[0],
                            bitdepth=16,
                            greyscale=True)
        writer.write(f, depth_map.tolist())


def depth_overlay(depth_path, img_path, out_path):
    orig_image = cv2.imread(img_path)
    depth_image = cv2.imread(depth_path)

    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    alpha = 0.6
    overlaid_image = cv2.addWeighted(orig_image, 1-alpha, depth_colored, alpha, 0)

    cv2.imwrite(out_path, overlaid_image)


def depth_overlay1(depth_path, img_path, out_path):
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.uint16)

    valid_mask = depth_map > 0
    v, u = np.where(valid_mask)

    depth = (65535 * depth_map[v, u] / 500)
    min_depth, max_depth = depth.min(), depth.max()
    depth_norm = ((max_depth - depth) / (max_depth - min_depth) * 255).astype(np.uint8)

    depth_gray = np.zeros_like(depth_map, dtype=np.uint8)
    depth_gray[v, u] = depth_norm

    color_map = cv2.applyColorMap(depth_gray, cv2.COLORMAP_JET)
    color_map_rgb = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

    for i in range(len(v)):
        color = tuple(map(int, color_map_rgb[v[i], u[i]]))
        center = (u[i], v[i])
        cv2.circle(orig_img, center, radius=5, color=color, thickness=-1)

    orig_image_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, orig_image_bgr)



def get_stats(pcd_path, width, height, in_mat, ex_mat):
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    count = 0
    depth_arr = []

    for point in points:
        x, y, z = point
        point_3d = np.array([x, y, z, 1])

        # world to camera
        point_cam = ex_mat @ point_3d  # (4x4) mult (4x1) = (4x1)
        # camera to pixel
        point_px = in_mat @ point_cam  # (3x4) mult (4x1) = (3x1)

        w = point_px[2]
        u, v = int(point_px[0] / w), int(point_px[1] / w)

        if 0 <= v < height and 0 <= u < width:
            count += 1
            depth_arr.append(w)

    return count, depth_arr


def get_stats1(pcd_path, width, height, in_mat, ex_mat):
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    count = 0
    depth_arr = []

    # Convert to homogeneous coordinates (N, 4)
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))

    points_cam = ex_mat @ points_h.T
    points_cam = points_cam.T

    # Camera to pixel transformation (3, 4) x (4, N) → (3, N)
    points_px = in_mat @ points_cam.T
    points_px = points_px.T

    # Normalize to get pixel coordinates
    w = points_px[:, 2]
    u = np.round(points_px[:, 0] / w).astype(int)
    v = np.round(points_px[:, 1] / w).astype(int)

    for i in range(len(u)):
        if (0 <= u[i]) & (u[i] < width) & (0 <= v[i]) & (v[i] < height):
            count += 1
            depth_arr.append(w[i])

    return count, depth_arr


def eliminate_offset(img1_path, img2_path, save_path_img1, save_path_img2):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    height, width, _ = img1.shape
    block_h = 3000
    
    cropped = img1[:block_h, :]
    result = cv2.matchTemplate(img2, cropped, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    _, match_top = max_loc
    crop_h = height - match_top

    cropped_img1 = img1[:crop_h, :]
    cropped_img2 = img2[match_top:, :]

    cv2.imwrite(save_path_img1, cropped_img1)
    cv2.imwrite(save_path_img2, cropped_img2)

    return match_top


def report_offset(img1_path, img2_path, name, block_h):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    height, width, _ = img1.shape

    cropped = img1[:block_h, :]
    result = cv2.matchTemplate(img2, cropped, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    _, match_top = max_loc
    report = f"{match_top} pixels for {name}"
    print(report)

    return report, match_top


def report_avg_offset(img1_path, img2_path, name, block_h=3000, step=100):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    offset_arr = []

    height, width, _ = img1.shape
    y = 0
    while y + block_h < height - step:
        cropped = img1[y:min(y+block_h, height), :]
        result = cv2.matchTemplate(img2, cropped, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        _, match_top = max_loc
        offset_arr.append(abs(y - match_top))
        print(y, abs(y - match_top))
        y += step

    avg_offset = statistics.mean(offset_arr)
    report = f"{avg_offset} pixels for {name}"
    print(report)
    return report, avg_offset


def report_error(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    height, width, _ = img1.shape

    target_block_height = 1000
    target_block = img1[:target_block_height, :]

    result = cv2.matchTemplate(img2, target_block, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)  # the top left corner of best matched area from right image
    _, y_best_match = max_loc  # the y coordinate of top left corner best match

    vertical_offset = y_best_match

    return vertical_offset


def find_best_match_px(target_px, px_list):
    target_px = np.array(target_px)
    px_list = np.array(px_list)

    diff = np.linalg.norm(px_list - target_px, axis=1)
    best_match_index = np.argmin(diff)

    return best_match_index


def pcd2disp_pn(pcd_path, left_img_path, in_ex_left, in_ex_right, out_path, size=(5472, 3648)):
    width = size[0]
    height = size[1]

    img = cv2.imread(left_img_path)

    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)  # Shape: (N, 3)

    # Convert points to homogeneous coordinates (N, 4)
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))  # (N, 4)

    # Project points to left camera
    left_points = (in_ex_left[1] @ points_h.T).T  # (N, 4)
    left_px = (in_ex_left[0] @ left_points.T).T  # (N, 3)
    wl = left_px[:, 2]
    ul = np.round(left_px[:, 0] / wl).astype(int)
    vl = np.round(left_px[:, 1] / wl).astype(int)

    # Project points to right camera
    right_points = (in_ex_right[1] @ points_h.T).T  # (N, 4)
    right_px = (in_ex_right[0] @ right_points.T).T  # (N, 3)
    wr = right_px[:, 2]
    ur = np.round(right_px[:, 0] / wr).astype(int)
    vr = np.round(right_px[:, 1] / wr).astype(int)

    # Filter points inside image bounds
    valid_mask = (ul < width) & (ur < width) & (vl < height) & (vr < height)
    valid_mask &= (ul >= 0) & (vl >= 0) & (ur >= 0) & (vr >= 0)

    ul = ul[valid_mask]
    vl = vl[valid_mask]
    ur = ur[valid_mask]
    wl = wl[valid_mask]

    disparity = ul - ur
    disp_map = np.zeros((height, width), dtype=np.int32)
    disp_map[vl, ul] = disparity

    depth_map = np.zeros((height, width), dtype=np.uint16)
    depth_map[vl, ul] = wl

    neg_disp_arr = []
    v_neg, u_neg = np.where(disp_map < 0)
    neg_disp_arr = depth_map[v_neg, u_neg].tolist()
    for i in range(len(u_neg)):
        center = u_neg[i], v_neg[i]
        cv2.circle(img, center, radius=10, color=(0, 0, 255), thickness=-1)

    pos_disp_arr = []
    v_pos, u_pos = np.where(disp_map > 0)
    pos_disp_arr = depth_map[v_pos, u_pos].tolist()
    for i in range(len(u_pos)):
        center = u_pos[i], v_pos[i]
        cv2.circle(img, center, radius=10, color=(255, 0, 0), thickness=-1)

    cv2.imwrite(out_path, img)

    return neg_disp_arr, pos_disp_arr


def pcd2disp(pcd_path, in_ex_left, in_ex_right, out_path, size=(5472, 3648)):
    width = size[0]
    height = size[1]

    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)  # Shape: (N, 3)

    # Convert points to homogeneous coordinates (N, 4)
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))  # (N, 4)

    # Project points to left camera
    left_points = (in_ex_left[1] @ points_h.T).T  # (N, 4)
    left_px = (in_ex_left[0] @ left_points.T).T  # (N, 3)
    wl = left_px[:, 2]
    ul = np.round(left_px[:, 0] / wl).astype(int)
    vl = np.round(left_px[:, 1] / wl).astype(int)

    # Project points to right camera
    right_points = (in_ex_right[1] @ points_h.T).T  # (N, 4)
    right_px = (in_ex_right[0] @ right_points.T).T  # (N, 3)
    wr = right_px[:, 2]
    ur = np.round(right_px[:, 0] / wr).astype(int)
    vr = np.round(right_px[:, 1] / wr).astype(int)

    # Filter points inside image bounds
    valid_bound_mask = (ul < width) & (ur < width) & (vl < height) & (vr < height)
    valid_bound_mask &= (ul >= 0) & (vl >= 0) & (ur >= 0) & (vr >= 0)

    ul = ul[valid_bound_mask]
    vl = vl[valid_bound_mask]
    ur = ur[valid_bound_mask]

    # Filter points with positive disparity
    disparity = ul - ur
    max_disp = np.max(disparity)
    min_disp = np.min(disparity)
    valid_disp_mask = disparity > 0
    ul = ul[valid_disp_mask]
    vl = vl[valid_disp_mask]
    disparity = disparity[valid_disp_mask].astype(np.float32)

    # Normalize disparity
    disp_norm = (disparity / 1000 * 65535).astype(np.uint16)

    # Fill disparity map and save
    disp_map = np.zeros((height, width), dtype=np.uint16)
    disp_map[vl, ul] = disp_norm

    cv2.imwrite(out_path, disp_map)

    return max_disp, min_disp
