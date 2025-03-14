from utils import gen_cross_map
from tqdm import tqdm
import os

img1_dir = "datasets/data/test_2_14/cross_test/cross_uni_rec/img1"
img2_dir = "datasets/data/test_2_14/cross_test/cross_uni_rec/img2"
out_dir = "datasets/data/test_2_14/cross_test/cross_out"

for item in tqdm(os.listdir(img1_dir)):
    item_name = os.path.splitext(item)[0]

    img1_path = os.path.join(img1_dir, f"{item_name}.png")
    img2_path = os.path.join(img2_dir, f"{item_name}.png")
    out_path = os.path.join(out_dir, f"{item_name}.png")

    gen_cross_map(img1_path, img2_path, out_path)
