"""
Script to convert nd2 file to frame tif files in bulk.
"""

import nd2reader
from skimage import io
from glob import glob
import os
from tqdm import trange
import numpy as np


def get_movie_frame(movie, frame_idx: int):
    """
    Given a movie and a frame, load the frame from the movie
    """
    movie.bundle_axes = ["y", "x", "c"]
    movie_frame = movie.get_frame(frame_idx)
    return np.array(movie_frame, dtype=np.uint16)


def convert_nd2_to_frame_tif(nd2, prefix, output_dir, max_frame=None):
    nd2_len = nd2.sizes["t"]
    for frame_idx in trange(nd2_len, desc=f"Converting {nd2_file}"):
        # for frame_idx in trange(221, 222, desc=f"Converting {nd2_file}"):
        output_file = f"{output_dir}/{prefix}_{frame_idx:04}_img.tif"
        movie_frame = get_movie_frame(nd2, frame_idx)
        movie_frame = np.moveaxis(movie_frame, 2, 0)
        io.imsave(output_file, movie_frame)
        if max_frame is not None and frame_idx >= max_frame:
            break


# Usage example
if __name__ == "__main__":
    # root_dir = "/Volumes/Extreme SSD/ZachML/CellType"
    # root_dir = "/Users/zachmaas/Desktop"
    # root_dir = "/Users/zachmaas/OneDrive - UCB-O365/202312 CyPose/"
    # output_dir = "/Users/zachmaas/builds/cyano_images/original_2ch"
    # output_dir = "/Users/zachmaas/builds/cyano_images/jian/ana_2"
    # root_dir = "/Users/zachmaas/OneDrive - UCB-O365/202312 CyPose/"
    # root_dir = "/Users/zachmaas/OneDrive - UCB-O365/202312 CyPose/"
    root_dir = "/Users/zachmaas/Desktop/AnaFiles/"
    output_dir = "/Users/zachmaas/builds/cyano_images/jian/better_ana"
    # Check for directories to exist
    if not os.path.isdir(root_dir):
        raise Exception(f"Root directory {root_dir} does not exist")
    if not os.path.isdir(output_dir):
        raise Exception(f"Output directory {output_dir} does not exist")
    files = glob(f"{root_dir}/*.nd2")
    files = (
        "/Users/zachmaas/Desktop/AnaFiles/20210709_Ana_-N_to_-N_channelbf,cy5,cfp,rfp_seq0000_0000.nd2",
    )
    max_frames = (50,)
    if len(files) == 0:
        raise Exception(f"No nd2 files found in root directory {root_dir}")
    # Max index to process
    for i, nd2_file in enumerate(files):
        prefix = os.path.basename(nd2_file).split("_")[0]
        # prefix = os.path.basename(nd2_file).split("_")[0]
        # prefix = "tabs_new"
        with nd2reader.ND2Reader(nd2_file) as nd2:
            convert_nd2_to_frame_tif(
                nd2, prefix, output_dir, max_frame=max_frames[i]
            )
