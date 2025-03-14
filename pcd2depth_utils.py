import open3d as o3d
import numpy as np
import png
import cv2


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
    valid_mask = (0 <= u) & (u < width) & (0 <= v) & (v < height)
    u, v, w = u[valid_mask], v[valid_mask], w[valid_mask]

    # Normalize depth values for better visualization
    max_depth = np.max(w)
    w_uint16 = (w / max_depth * 65535).astype(np.uint16)

    depth_map = np.full((height, width), 65535, dtype=np.uint16)

    for i in range(len(u)):
        cv2.circle(depth_map, (u[i], v[i]), 5, int(w_uint16[i]), thickness=-1)
        depth_map[v[i], u[i]] = np.minimum(w_uint16[i], depth_map[v[i], u[i]])

    depth_map[depth_map == 65535] = 0
    max_depth = np.max(depth_map)
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
    depth_image = cv2.addWeighted(orig_image, 1-alpha, depth_colored, alpha, 0)

    cv2.imwrite(out_path, depth_image)
