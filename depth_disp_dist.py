import open3d as o3d
import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt


cam3_ex_mat = [[
                        0.999613,
                        0.0194676,
                        0.0115788,
                        0.250893
                    ],
                    [
                        0.0104086,
                        0.05054,
                        -0.998686,
                        -0.0072955
                    ],
                    [
                        -0.0198467,
                        0.998423,
                        0.0498983,
                        0.0234565
                    ],
                    [
                        0,
                        0,
                        0,
                        1
                    ]]

cam2_ex_mat = [[
                        0.999941,
                        0.00929654,
                        0.00586799,
                        -0.25716
                    ],
                    [
                        0.00562645,
                        0.00627855,
                        -0.999981,
                        -0.00087487
                    ],
                    [
                        -0.00915342,
                        0.999811,
                        0.00578637,
                        0.0207079
                    ],
                    [
                        0,
                        0,
                        0,
                        1
                    ]]

cam3_in_mat = [[30904.3, 0, 2446.95, 0],
               [0, 30142.5, 583.125, 0],
               [0, 0, 1.0, 0]]

cam2_in_mat = [[30895, 0, 2831.44, 0],
               [0, 29602.9, 1866.63, 0],
               [0, 0, 1.0, 0]]

in_ex_left = cam3_in_mat, cam3_ex_mat
in_ex_right = cam2_in_mat, cam2_ex_mat


def pcd2disp_dict(pcd_path, in_ex_left, in_ex_right, size=(5472, 3648)):
    width = size[0]
    height = size[1]

    depth_disp = {}

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

    depth = wl[valid_bound_mask].astype(int)

    for d, disp in zip(depth, disparity):
        if d not in depth_disp:
            depth_disp[d] = [disp]
        else:
            depth_disp[d].append(disp)

    return depth_disp


pcd_dir = "datasets/test_5_5/in/lidar"
depth_disp = {}

for item in os.listdir(pcd_dir):
    pcd_path = os.path.join(pcd_dir, item)
    cur_depth_disp = pcd2disp_dict(pcd_path, in_ex_left, in_ex_right)

    for k, v in cur_depth_disp.items():
        if k not in depth_disp:
            depth_disp[k] = v.copy()
        else:
            depth_disp[k].extend(v)

# Your bins: depth from 1 to 500 (inclusive)
bins = np.arange(0, 501)  # [1, 2, ..., 500]

# Store average disparity for each depth bin
avg_disparities = []

for depth in bins:
    if depth in depth_disp and len(depth_disp[depth]) > 0:
        avg_disp = np.mean(depth_disp[depth])
    else:
        avg_disp = 0  # or 0, depending on your preference
    avg_disparities.append(avg_disp)

# Convert to numpy array
avg_disparities = np.array(avg_disparities)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(bins, avg_disparities, label="Avg Disparity by Depth")
plt.xlabel("Depth")
plt.ylabel("Average Disparity")
plt.title("Average Disparity vs Depth")
plt.xlim(0, 500)
plt.savefig("datasets/test_5_5/in/avg_disparity_by_depth.png", dpi=300)
