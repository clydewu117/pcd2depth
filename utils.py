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
    points = np.asarray(pcd.points)

    depth_map = np.full((height, width), np.inf)

    radius = 5

    count = 0

    for point in points:
        x, y, z = point
        point_3d = np.array([x, y, z, 1])

        # world to camera
        point_cam = ex_mat @ point_3d  # (4x4) mult (4x1) = (4x1)
        # camera to pixel
        point_px = in_mat @ point_cam  # (3x4) mult (4x1) = (3x1)

        w = point_px[2]
        u, v = np.round(point_px[0] / w).astype(int), np.round(point_px[1] / w).astype(int)

        if 0 <= v < height and 0 <= u < width:
            mask = np.zeros_like(depth_map, dtype=np.uint8)
            cv2.circle(mask, (u, v), radius, 1, thickness=-1)
            mask_indices = mask == 1
            depth_map[mask_indices] = np.minimum(depth_map[mask_indices], w)
            count += 1

    depth_map[np.isinf(depth_map)] = 0
    max_depth = np.max(depth_map)
    non_zero_mask = depth_map > 0
    depth_map[non_zero_mask] = max_depth - depth_map[non_zero_mask]
    depth_map_uint16 = (depth_map * 256).astype(np.uint16)

    with open(out_path, 'wb') as f:
        writer = png.Writer(width=depth_map_uint16.shape[1],
                            height=depth_map_uint16.shape[0],
                            bitdepth=16,
                            greyscale=True)
        writer.write(f, depth_map_uint16.tolist())

    print(f"{count} depth points within the range")
    print(f"Depth map saved to {out_path}")


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
        cv2.circle(depth_map, (u[i], v[i]), 5, int(w_uint16[i]), thickness=-1)
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
    print(match_top)
    print(crop_h)

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


def report_misalignment(img_path, depth_path):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    depth = cv2.imread(depth_path)

    edges_img = cv2.Canny(img_gray, 100, 120)
    edges_depth = cv2.Canny(depth, 100, 200)

    intersection = np.logical_and(edges_img, edges_depth).sum()
    union = np.logical_or(edges_img, edges_depth).sum()
    iou = intersection / union

    print(iou)


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


def gen_cross_map(img1_path, img2_path, out_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

    cross_map = cv2.subtract(img1, img2)

    cv2.imwrite(out_path, cross_map)


# def find_min_disp(pcd_path, ex_mat1, in_mat1, ex_mat2, in_mat2, height, width):
#     pcd = o3d.io.read_point_cloud(pcd_path)
#     points = np.asarray(pcd.points)
#
#     disp_arr = []
#     w1_arr = []
#     w2_arr = []
#
#     for point in points:
#         x, y, z = point
#         point_3d = np.array([x, y, z, 1])
#
#         point_cam1 = ex_mat1 @ point_3d  # (4x4) mult (4x1) = (4x1)
#         point_px1 = in_mat1 @ point_cam1  # (3x4) mult (4x1) = (3x1)
#         w1 = point_px1[2]
#         u1, v1 = np.round(point_px1[0] / w1).astype(int), np.round(point_px1[1] / w1).astype(int)
#
#         point_cam2 = ex_mat2 @ point_3d  # (4x4) mult (4x1) = (4x1)
#         point_px2 = in_mat2 @ point_cam2  # (3x4) mult (4x1) = (3x1)
#         w2 = point_px2[2]
#         u2, v2 = np.round(point_px2[0] / w2).astype(int), np.round(point_px2[1] / w2).astype(int)
#
#         if v1 < height and v2 < height and u1 < width and u2 < width:
#             disp_arr.append(np.linalg.norm(np.array([u1, v1]) - np.array([u2, v2])))
#             w1_arr.append(w1)
#             w2_arr.append(w2)
#
#     min_disp = min(disp_arr)
#     i = disp_arr.index(min_disp)
#     w1_depth = w1_arr[i]
#     w2_depth = w2_arr[i]
#
#     return min_disp, w1_depth, w2_depth


def find_min_disp(pcd_path, ex_mat1, in_mat1, ex_mat2, in_mat2, height, width):
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)  # Shape: (N, 3)

    # Convert points to homogeneous coordinates (N, 4)
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))  # (N, 4)

    # Project points to first camera
    cam1_points = (ex_mat1 @ points_h.T).T  # (N, 4)
    px1 = (in_mat1 @ cam1_points.T).T  # (N, 3)
    w1 = px1[:, 2]
    u1 = np.round(px1[:, 0] / w1).astype(int)
    v1 = np.round(px1[:, 1] / w1).astype(int)

    # Project points to second camera
    cam2_points = (ex_mat2 @ points_h.T).T  # (N, 4)
    px2 = (in_mat2 @ cam2_points.T).T  # (N, 3)
    w2 = px2[:, 2]
    u2 = np.round(px2[:, 0] / w2).astype(int)
    v2 = np.round(px2[:, 1] / w2).astype(int)

    # Filter points inside image bounds
    valid_mask = (v1 < height) & (v2 < height) & (u1 < width) & (u2 < width)

    # Compute disparity only for valid points
    disp_arr = np.linalg.norm(np.vstack((u1, v1)).T - np.vstack((u2, v2)).T, axis=1)
    disp_arr = disp_arr[valid_mask]
    w1_valid = w1[valid_mask]
    w2_valid = w2[valid_mask]

    # Get minimum disparity and corresponding depths
    min_idx = np.argmin(disp_arr)
    min_disp = disp_arr[min_idx]
    w1_depth = w1_valid[min_idx]
    w2_depth = w2_valid[min_idx]

    return min_disp, w1_depth, w2_depth
