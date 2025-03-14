import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.spatial import KDTree


depth_path = "datasets/data/test_remove/62_depth.png"
depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).copy()
depth_points = np.column_stack(np.where(depth_map > 0))
kdtree = KDTree(depth_points)

updated_depth_path = "datasets/data/test_remove/updated.png"

removed_px = []


def on_click(event):

    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)

        print(f"Clicked at: ({x}, {y})")
        dist, index = kdtree.query([y, x])
        nearest_point = depth_points[index][::-1]
        removed_px.append(nearest_point)
        print(f"Will remove: {nearest_point}")


def on_key(event):

    global removed_px

    if event.key == 'g':  # Press 'G' to terminate the program
        print("Terminating the program")
        print("Final list of removed points:", removed_px)
        plt.close()

    elif event.key == 'h':  # Press 'H' to remove the most recent clicked coordinate
        if removed_px:
            removed_point = removed_px.pop()
            print(f"Undo last clicked point: {removed_point}")


image_path = "datasets/data/test_remove/62_depth_img.png"
img = plt.imread(image_path)

fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title("Click on the image | Press G to terminate | Press H to undo")

fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()

log_path = "datasets/data/test_remove/log.txt"

with open(log_path, 'a') as file:
    for px in removed_px:
        cv2.circle(depth_map, px, 10, 0, thickness=-1)
        file.write(f"{px}\n")

cv2.imwrite(updated_depth_path, depth_map)
