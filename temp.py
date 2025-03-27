from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = YOLO("yolov8s-seg.pt")

# Read image
img_path = "datasets/data/test_2_14/in/cam3_img/192.png"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = model(img)
result = results[0]

masks = result.masks
boxes = result.boxes
names = result.names

mask_arr = masks.data.cpu().numpy()
class_ids = boxes.cls.cpu().numpy().astype(int)
confidences = boxes.conf.cpu().numpy()

for i, mask in enumerate(mask_arr):
    class_name = names[class_ids[i]]
    confidence = confidences[i]
    plt.imshow(mask, cmap="gray")
    plt.title(f"Mask {i}: {class_name} ({confidence:.2f})")
    plt.axis("off")
    plt.show()
