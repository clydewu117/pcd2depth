import open3d as o3d
import numpy as np
import cv2
import png

pcd = o3d.io.read_point_cloud("datasets/data/lidar/0.pcd")

# o3d.visualization.draw_geometries([pcd])

points = np.asarray(pcd.points)

# define camera intrinsic parameters
width = 5472
height = 3648
fx = 31470.1  # x focal length
fy = 20825  # y focal length
cx, cy = width / 2, height / 2  # principal point

# intrinsic matrix
in_mat = np.array([[fx, 0, cx, 0],
                   [0, fy, cy, 0],
                   [0, 0, 1, 0]])

# extrinsic matrix
ex_mat = np.array([[0.999933899272713, -0.003245217941172, 0.0111, -0.217316],
                   [0.010881133852003, -0.0108, -0.9999, -0.00038],
                   [0.003544976558669, 0.9998, -0.0112, 0.2076],
                   [0, 0, 0, 1]])

# initialize empty depth map
depth_map = np.full((height, width), np.inf)

# project each point
for point in points:
    x, y, z = point
    point_3d = np.array([x, y, z, 1])

    # project to 2D, ignore the points behind the camera
    if z > 0:
        # world to camera
        point_cam = ex_mat @ point_3d  # (4x4) mult (4x1) = (4x1)
        # camera to pixel
        point_px = in_mat @ point_cam  # (3x4) mult (4x1) = (3x1)

        w = point_px[2]
        u, v = int(point_px[0]/w), int(point_px[1]/w)
        print(u, v, w)
        if 0 <= v < height and 0 <= u < width:
            depth_map[v, u] = w

# convert depth values of inf to 0 for better visualization
depth_map[np.isinf(depth_map)] = 0

# Scale the depth map to match KITTI’s format (depth values * 256) and convert to uint16
depth_map_uint16 = (depth_map * 256).astype(np.uint16)

# Save the depth map as a 16-bit PNG file
output_path = "test_depth.png"
with open(output_path, 'wb') as f:
    writer = png.Writer(width=depth_map_uint16.shape[1],
                        height=depth_map_uint16.shape[0],
                        bitdepth=16,
                        greyscale=True)
    writer.write(f, depth_map_uint16.tolist())

print(f"Depth map saved to {output_path}")
