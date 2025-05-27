import os
import shutil

dataset_dir = "datasets/test_5_5/uni_mid_half"
out_dir = "datasets/test_5_5/split"
folders = ["cam2_img", "cam3_img", "disp"]

# how to split
def get_split(filename):
    num = int(filename.replace(".png", ""))
    if 0 <= num <= 54:
        return "test"  # closest to intersection
    elif 55 <= num <= 81:
        return "val"    # mid distance
    elif 82 <= num <= 163:
        return "train"   # mid + far
    else:
        return None

splits = ["train", "val", "test"]
for split in splits:
    for folder in folders:
        os.makedirs(os.path.join(out_dir, split, folder), exist_ok=True)

split_lists = {s: [] for s in splits}

file_list = sorted(f for f in os.listdir(os.path.join(dataset_dir, "cam2_img")) if f.endswith(".png"))

for fname in file_list:
    split = get_split(fname)
    if split is None:
        continue

    for folder in folders:
        src = os.path.join(dataset_dir, folder, fname)
        dst = os.path.join(out_dir, split, folder, fname)
        shutil.copyfile(src, dst)

    split_lists[split].append(fname)

# log
for split in splits:
    txt_path = os.path.join(out_dir, f"{split}.txt")
    with open(txt_path, "w") as f:
        for fname in split_lists[split]:
            f.write(fname + "\n")

print("split done")
for split in splits:
    print(f"{split}: {len(split_lists[split])} images")
