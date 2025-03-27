from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm
import json

img_dir = "datasets/data/test_2_14/in/cam3_img"
model = YOLO("yolov8s-seg.pt")

result_dict = {}

for item in tqdm(os.listdir(img_dir)):
    img_path = os.path.join(img_dir, item)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model(img)[0]
    class_ids = results.boxes.cls.int().tolist()
    class_names = [model.names[c] for c in class_ids]

    result_dict[item] = class_names

with open("segment_results.json", "w") as f:
    json.dump(result_dict, f, indent=4)
