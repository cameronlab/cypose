# diff_masks.py --- Calculate the difference between two masks
#
# Filename: diff_masks.py
# Author: Zach Maas
# Created: Wed Dec 20 12:29:16 2023 (-0700)
#
#

# Commentary:
#
# This file contains code to calculate the difference between two
# masks. This is useful for benchmarking different models.
#

# Code:

import skimage
import numpy as np
from tqdm import tqdm

# Define the folder path
folder_path = "/Users/zachmaas/Desktop"
# file_1 = "masks_2.tif"
# file_2 = "masks_3.tif"
file_1 = "our_mask.tif"
file_2 = "their_mask.tif"

# Read the two files
img_1 = skimage.io.imread(f"{folder_path}/{file_1}")
img_2 = skimage.io.imread(f"{folder_path}/{file_2}")

# Calculate the difference
diff = img_2 - img_1

# Calculate the number of pixels that are different
num_different = (diff != 0).sum()
# Calculate the total number of pixels
num_total = diff.size
# Calculate the percentage of pixels that are different
percent_different = num_different / num_total
# Print the percentage
print(f"{(1 - percent_different) * 100}% similarity")

# Combine file1, file2, and the difference into 3 channels
combined = np.stack((img_1, img_2, diff), axis=1)
# Reformat for FIJI
combined = np.transpose(combined, (0, 2, 3, 1))
# Save the combined image
skimage.io.imsave(f"{folder_path}/mask_difference.tif", combined)
# Save the difference
skimage.io.imsave(f"{folder_path}/diff.tif", diff)

# if __name__ == "__main__":
#     return 0

#
# diff_masks.py ends here
