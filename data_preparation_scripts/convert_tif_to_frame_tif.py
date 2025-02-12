"""
Script to convert tif file to frame tif files in bulk.
"""

from skimage import io
from glob import glob
import os
from tqdm import trange, tqdm
import numpy as np


def convert_tif_to_frame_tif(tif, prefix, output_dir, max_frame=None):
    tif_handle = io.imread(tif)
    for frame_idx, img in tqdm(enumerate(tif_handle), desc=f"Converting {tif}"):
        output_file = f"{output_dir}/{prefix}_{frame_idx:04}_masks.tif"
        movie_frame = np.array(img, dtype=np.uint16)
        io.imsave(output_file, movie_frame)
        if max_frame is not None and frame_idx >= max_frame:
            break


# Usage example
if __name__ == "__main__":
    # root_dir = "/Users/zachmaas/Desktop/AnaFiles/"
    output_dir = "/Users/zachmaas/builds/cyano_images/jian/better_ana"
    # Check for directories to exist
    if not os.path.isdir(output_dir):
        raise Exception(f"Output directory {output_dir} does not exist")
    # prefix = "tabs_new"
    # files = glob(f"{root_dir}/*.tif")
    files = (
        "/Users/zachmaas/Desktop/AnaFiles/20210709_Ana_-N_to_-N_channelbf,cy5,cfp,rfp_seq0000_0000_Mask.tif",
    )
    for tif_file in files:
        prefix = os.path.basename(tif_file).split("_")[0]
        convert_tif_to_frame_tif(tif_file, prefix, output_dir)
