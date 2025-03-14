import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
import cv2
import statistics

left_image_path = "datasets/data/test/cropped_image_left.png"
right_image_path = "datasets/data/test/cropped_image_right.png"

# import the images
left_image = Image.open(left_image_path)
right_image = Image.open(right_image_path)

# convert to numpy array
left_image = np.array(left_image)
right_image = np.array(right_image)

print("Left image shape:", left_image.shape)
print("Right image shape:", right_image.shape)


def calc_vertical_offset(left_image, right_image, step=100, target_block_height=3000):
    offset_arr = []

    height, width, _ = left_image.shape
    y = 0
    while y + target_block_height < height:
        cropped = left_image[y:min(y + target_block_height, height), :]
        result = cv2.matchTemplate(right_image, cropped, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        _, match_top = max_loc
        offset_arr.append(abs(y - match_top))
        print(y, abs(y - match_top))
        y += step

    avg_offset = statistics.mean(offset_arr)
    return avg_offset


# for each pixel in the left image, find the corresponding pixel in the right image (but only in the same row)
# for each pixel in the left image, find the corresponding pixel in the right image (but only in the same row)
def find_corresponding_pixel(left_image, right_image, max_disparity=500):
    # get the height and width of the images
    height, width, _ = left_image.shape

    # create an empty array to store the corresponding pixels
    corresponding_pixels = np.zeros((height, width, 3), dtype=np.uint8)

    # create an empty array to store the pixel differences with the pixel in the left image and the corresponding pixel
    pixel_differences_map = np.zeros((height, width))

    # loop through each pixel in the left image
    for x in tqdm(range(width)):
        # get the pixel value from the left image
        left_pixel = left_image[:, x]

        best_match_err = np.empty(height)
        best_match_err.fill(999999)

        # find the best match pixel in the right image (same row, from 0 to x)
        if x < max_disparity:
            start_idx = 0
        else:
            start_idx = x - max_disparity

        for i in range(start_idx, x):
            # get the pixel value from the right image
            right_pixel = right_image[:, i]

            # calculate the difference between the left pixel and the right pixel
            diff = np.sum(np.abs(left_pixel - right_pixel), axis=1)
            # if the difference is smaller than the best match, update the best match

            # if diff < best_match_err:
            #    best_match_err = diff
            #    #best_match_index = i
            # print(best_match_err)
            # print(diff)
            best_match_err = np.minimum(best_match_err, diff)

        pixel_differences_map[:, x] = best_match_err

    return pixel_differences_map

# crop top-left 100 pixels from the left and right images


pixel_differences_map = find_corresponding_pixel(left_image, right_image)

# print statistics
print("Pixel differences map shape:", pixel_differences_map.shape)
print("Pixel differences map min:", np.min(pixel_differences_map))
print("Pixel differences map max:", np.max(pixel_differences_map))
print("Pixel differences map mean:", np.mean(pixel_differences_map))
#print("Pixel differences map std:", np.std(pixel_differences_map))
print("Pixel differences map median:", np.median(pixel_differences_map))
#print("Pixel differences map 95th percentile:", np.percentile(pixel_differences_map, 95))
#print("Pixel differences map 99th percentile:", np.percentile(pixel_differences_map, 99))

# replace 999999 with 0
pixel_differences_map[pixel_differences_map == 999999] = 0

# plot left image and pixel differences map
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(left_image)
plt.title("Left Image")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(right_image)
plt.title("Right Image")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(pixel_differences_map, cmap="gray")
plt.title("Pixel Differences Map")
plt.axis("off")
plt.tight_layout()
plt.show()

# calculate the vertical offset

vertical_offset = calc_vertical_offset(left_image, right_image)
print("Vertical offset:", vertical_offset)
