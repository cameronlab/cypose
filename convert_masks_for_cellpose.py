import cv2
import skimage.io as io
import skimage.util as util
import glob
from tqdm import tqdm
import numpy as np

# Define the folder path
folder_path = "/Users/zachmaas/builds/cyano_images/jian/better_ana"

# Get a list of all image files matching the pattern
image_files = glob.glob(folder_path + "/*_masks.tif")

# Process each image file
for image_file in tqdm(image_files):
    # Read the image
    # img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    img = io.imread(image_file).astype("uint8")

    # Invert if needed
    invert = False
    if invert:
        img = util.invert(img)

    # Label the connected components
    labeled = cv2.connectedComponents(img)[1]

    # Create the new file name with the "_labeled.tif" suffix
    new_file_name = image_file.replace("_masks.tif", "_labeled.tif")

    # Save the labeled image
    cv2.imwrite(new_file_name, labeled)
