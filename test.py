import os
import matplotlib.pyplot as plt
import numpy as np
import re
import json

with open("segment_results.json", "r") as f:
    loaded_dict = json.load(f)

lengths = [len(v) for k, v in loaded_dict.items()]

max_len = max(lengths)
min_len = min(lengths)

plt.hist(lengths, bins=range(min_len, max_len), edgecolor='black')
plt.title("segments / frames", fontsize=16)
plt.xlabel("number of segments", fontsize=16)
plt.ylabel("number of frames", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
