import os
import cv2
from tqdm import tqdm


def crop_lower_half(input_path, output_path=None):
    image = cv2.imread(input_path)
    height = image.shape[0]
    lower_half = image[148:, :]

    if output_path:
        cv2.imwrite(output_path, lower_half)

    return lower_half


data_dir = "datasets/data/test_2_9/out/cam3_depth"

out_dir = "datasets/data/test_2_9/out/cam3_depth1"

for item in tqdm(os.listdir(data_dir)):
    depth_path = os.path.join(data_dir, item)
    out_path = os.path.join(out_dir, item)

    crop_lower_half(depth_path, out_path)
