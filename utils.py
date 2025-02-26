import os.path

import open3d as o3d
import numpy as np
import cv2
import png
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


def depth_overlay(depth_path, img_path, out_path):
    orig_image = cv2.imread(img_path)
    depth_image = cv2.imread(depth_path)

    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    alpha = 0.6
    overlaid_image = cv2.addWeighted(orig_image, 1-alpha, depth_colored, alpha, 0)

    cv2.imwrite(out_path, overlaid_image)

    print(f"Image with depth saved to {out_path}")


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


# def report_noise(pcd_path, left_img_path, right_img_path, width, height, matrices_cam2, matrices_cam3):
#     pcd = o3d.io.read_point_cloud(pcd_path)
#     points = np.asarray(pcd.points)
#
#     left_img = Image.open(left_img_path)
#     right_img = Image.open(right_img_path)
#
#     rgb_diffs = []
#     count = 0
#
#     for point in points:
#         x, y, z = point
#         point_3d = np.array([x, y, z, 1])
#
#         # world to camera2
#         point_cam2 = matrices_cam2[0] @ point_3d  # (4x4) mult (4x1) = (4x1)
#         # camera to pixel2
#         point_px2 = matrices_cam2[1] @ point_cam2  # (3x4) mult (4x1) = (3x1)
#
#         # world to camera3
#         point_cam3 = matrices_cam3[0] @ point_3d  # (4x4) mult (4x1) = (4x1)
#         # camera to pixel3
#         point_px3 = matrices_cam3[1] @ point_cam3  # (3x4) mult (4x1) = (3x1)
#
#         w2 = point_px2[2]
#         u2, v2 = int(point_px2[0] / w2), int(point_px2[1] / w2)
#
#         w3 = point_px3[2]
#         u3, v3 = int(point_px3[0] / w3), int(point_px3[1] / w3)
#
#         if 0 <= v2 < height and 0 <= u2 < width and 0 <= v3 < height and 0 <= u3 < width:
#             loc_left = (u2, v2)
#             loc_right = (u3, v3)
#             rgb_diff = compare_rgb(loc_left, loc_right, left_img, right_img)
#             print(f"rgb diff: {rgb_diff}")
#             rgb_diffs.append(rgb_diff)
#
#         count += 1
#         # print(f"processing {count}/{len(points)}")
#
#     return rgb_diffs
#
#
# def compare_rgb(loc_left, loc_right, left_img, right_img):
#
#     r_left, g_left, b_left = left_img.getpixel(loc_left)
#     r_right, g_right, b_right = right_img.getpixel(loc_right)
#
#     return ((r_left-r_right)**2+(g_left-g_right)**2+(b_left-b_right)**2) / 3

def eliminate_offset(img1_path, img2_path, save_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    height, width, _ = img1.shape
    block_h = 1000
    
    cropped = img1[:block_h, :]
    result = cv2.matchTemplate(img2, cropped, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    _, match_top = max_loc
    crop_h = height - match_top
    print(match_top)
    print(crop_h)

    cropped_img1 = img1[:crop_h, :]
    cropped_img2 = img2[match_top:, :]

    save_path_img1 = os.path.join(save_path, 'cropped_image_left.png')
    save_path_img2 = os.path.join(save_path, 'cropped_image_right.png')
    cv2.imwrite(save_path_img1, cropped_img1)
    cv2.imwrite(save_path_img2, cropped_img2)


def report_offset(img1_path, img2_path, name):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    height, width, _ = img1.shape
    block_h = 1000

    cropped = img1[:block_h, :]
    result = cv2.matchTemplate(img2, cropped, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    _, match_top = max_loc
    report = f"{match_top} pixels for {name}"
    print(report)

    return report, match_top


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


def report_error(img1_path, img2_path, save_dir):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    pre_error_map_path = os.path.join(save_dir, "pre_error_map.png")
    post_error_map_path = os.path.join(save_dir, "post_error_map.png")

    height, width, _ = img1.shape

    target_block_height = 1000
    target_block = img1[:target_block_height, :]

    result = cv2.matchTemplate(img2, target_block, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)  # the top left corner of best matched area from right image
    _, y_best_match = max_loc  # the y coordinate of top left corner best match

    vertical_offset = y_best_match
