import open3d as o3d
import numpy as np
import cv2
import png


def pcd2depth(pcd_path, width, height, in_mat, ex_mat, out_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    depth_map = np.full((height, width), np.inf)

    radius = 5

    count = 0

    for point in points:
        x, y, z = point
        point_3d = np.array([x, y, z, 1])

        # project to 2D, ignore the points behind the camera
        # world to camera
        point_cam = ex_mat @ point_3d  # (4x4) mult (4x1) = (4x1)
        # camera to pixel
        point_px = in_mat @ point_cam  # (3x4) mult (4x1) = (3x1)

        w = point_px[2]
        u, v = int(point_px[0] / w), int(point_px[1] / w)

        if 0 <= v < height and 0 <= u < width:
            mask = np.zeros_like(depth_map, dtype=np.uint8)
            cv2.circle(mask, (u, v), radius, 1, thickness=-1)
            mask_indices = mask == 1
            depth_map[mask_indices] = np.minimum(depth_map[mask_indices], w)
            count += 1

    depth_map[np.isinf(depth_map)] = 0
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
