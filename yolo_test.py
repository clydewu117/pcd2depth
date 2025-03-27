from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt


model = YOLO("yolov8s-seg.pt")

# Read image
img_path = "datasets/data/test_2_14/in/cam3_img/192.png"
results = model(img_path)
result = results[0]

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Read depth map
depth_path = "datasets/data/test_2_14/out_new_ex/cam3_depth/192_depth.png"
depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

# Get all segmentation masks (N, H, W)
masks = result.masks.data.cpu().numpy()  # Convert to NumPy

# Set up plot
fig, axes = plt.subplots(3, len(masks) + 1, figsize=(15, 5))
axes[1, 0].axis("off")
axes[2, 0].axis("off")
fig.subplots_adjust(hspace=0.5)

# Scaling factor
scale_factor = 1.2

# Show original image at [0, 0]
axes[0, 0].imshow(img)
axes[0, 0].set_title("Original Image")

# Create list to store depth points in each mask
depth_per_mask = [[] for i in range(len(masks))]
depth_per_ring_mask = [[] for j in range(len(masks))]

# Process each segmentation mask
for i, mask in enumerate(masks):
    binary_mask = (mask > 0.5).astype(np.uint8)  # Convert to binary (0 or 1)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Display mask
    axes[0, i + 1].imshow(binary_mask, cmap="gray")
    axes[0, i + 1].set_title(f"Mask {i + 1}")

    # Create an empty canvas
    scaled_mask = np.zeros_like(binary_mask)

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue  # Skip empty contours

        # Compute centroid
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Scale contour points
        scaled_contour = (contour - [cx, cy]) * scale_factor + [cx, cy]
        scaled_contour = scaled_contour.astype(np.int32)

        # Draw scaled contour on the empty mask
        cv2.drawContours(scaled_mask, [scaled_contour], -1, 1, thickness=cv2.FILLED)

        # Draw red contour outline on plot
        axes[1, i + 1].imshow(scaled_mask, cmap="gray")

    axes[1, i + 1].set_title(f"Scaled Mask {i + 1}")

    ring_mask = ((scaled_mask > 0) & (binary_mask == 0)).astype(np.uint8)
    axes[2, i + 1].imshow(ring_mask, cmap="gray")
    axes[2, i + 1].set_title(f"Ring Mask {i + 1}")

    # Extract depth points located within original mask
    h, w = depth_map.shape[:2]
    binary_mask_resized = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    depth_values = depth_map[binary_mask_resized > 0]
    depth_values = depth_values[depth_values > 0]
    depth_values = 500 - depth_values / 65535 * 500
    depth_per_mask[i] = depth_values.tolist()

    # Extract depth points located within ring mask
    ring_mask_resized = cv2.resize(ring_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    depth_values_ring = depth_map[ring_mask_resized > 0]
    depth_values_ring = depth_values_ring[depth_values_ring > 0]
    depth_values_ring = 500 - depth_values_ring / 65535 * 500
    depth_per_ring_mask[i] = depth_values_ring.tolist()

plt.show()

for i in range(len(depth_per_mask)):
    print(f"mask {i+1} num of depth in the original mask: {len(depth_per_mask[i])}")
    print(f"first 10 depth values in the original mask: {depth_per_mask[i][:10]}")
    print(f"mask {i+1} num of depth in the ring mask: {len(depth_per_ring_mask[i])}")
    print(f"first 10 depth values in the ring mask: {depth_per_ring_mask[i][:10]}")

num_bins = 50

class_ids = result.boxes.cls.cpu().numpy().astype(int)
confidences = result.boxes.conf.cpu().numpy()
class_names = [model.names[cid] for cid in class_ids]

for i in range(len(depth_per_mask)):
    if len(depth_per_mask[i]) == 0:
        continue

    fig, ax = plt.subplots(figsize=(8, 5))

    min_value = min(depth_per_mask[i])
    max_value = max(depth_per_mask[i])
    bins = np.linspace(min_value, max_value, num_bins + 1)

    ax.hist(depth_per_mask[i], bins=bins, edgecolor='blue', alpha=0.5, label="Points in Original Mask")
    ax.hist(depth_per_ring_mask[i], bins=bins, edgecolor='red', alpha=0.5, label="Points in Ring Mask")
    ax.set_xlabel("Depth Value", fontsize=16)
    ax.set_ylabel("Number of Depth Points", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.legend(fontsize=16)

    class_name = class_names[i]
    confidence = confidences[i]
    plt.title(f"Mask {i + 1}: {class_name} ({confidence:.2f})", fontsize=16)

    plt.show()
