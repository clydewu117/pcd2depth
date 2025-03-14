import os
import matplotlib.pyplot as plt
import numpy as np
import re

cam2_sc_path = "datasets/data/test_2_14/in/cam2_sc_log.txt"
cam3_sc_path = "datasets/data/test_2_14/in/cam3_sc_log.txt"

# Read the log file
with open(cam3_sc_path, "r") as f:
    lines = f.readlines()

# Extract the numerical part and sort
sorted_lines = sorted(lines, key=lambda x: int(re.search(r"score for (\d+)\.png", x).group(1)))

# Write the sorted lines back
with open("datasets/data/test_2_14/in/cam3_sc_log1.txt", "w") as f:
    f.writelines(sorted_lines)

# Print the sorted output
print("".join(sorted_lines))
