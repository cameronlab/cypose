# run_segmentation.py --- Run segmentation on a tiff movie
#
# Filename: run_segmentation.py
# Author: Zach Maas
# Created: Tue Nov 14 10:23:24 2023 (-0700)
#
#

# Commentary:
#
#
# This file contains code to run cellpose segmentation on a tiff
# movie, using our custom finetuned model that's adapted to [idk
# cyanobacteria name spelling] 7002. This script is adapted to take in
# a movie, convert the z-stack to single images as required by
# cellpose, and then run segmentation. Output will then be coerced
# from the cellpose output format (1 channel per cell) to a single
# channel greyscale image, and saved as a file named ".seg".
#
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <https://www.gnu.org/licenses/>.
#
#

# Code:

from nd2reader import ND2Reader  # ND2 file reading
import tifffile  # Tiff file writing
import argparse  # Command line arguments
from cellpose import models  # Cellpose
from tqdm import tqdm  # Progress bar
import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run segmentation on a tiff movie")
parser.add_argument(
    "--input_file",
    metavar="input_file",
    type=str,
    nargs=1,
    required=True,
    help="The input file to run segmentation on",
)
parser.add_argument(
    "--output_file",
    metavar="output_file",
    type=str,
    nargs=1,
    required=True,
    help="The output file to save the segmentation to",
)
parser.add_argument(
    "--model",
    metavar="model",
    type=str,
    nargs=1,
    required=True,
    help="The model to use for segmentation",
)
parser.add_argument("--gpu", action=argparse.BooleanOptionalAction)
parser.add_argument(
    "--start_frame",
    metavar="start_frame",
    type=int,
    nargs="?",
    help="The frame to start segmentation on",
)
parser.add_argument(
    "--end_frame",
    metavar="end_frame",
    type=int,
    nargs="?",
    help="The frame to end segmentation on",
)
parser.add_argument(
    "--debug",
    action=argparse.BooleanOptionalAction,
    help="Run in debug mode, which will save additional output files",
)

# Check for required arguments
if len(parser.parse_args().input_file) == 0:
    print("No input file provided")
    exit(1)
if len(parser.parse_args().output_file) == 0:
    print("No output file provided")
    exit(1)
if len(parser.parse_args().model) == 0:
    print("No model provided")
    exit(1)
# Parse the arguments
args = parser.parse_args()
input_file = args.input_file[0]  # Check array indexing? Should just be a string
output_file = args.output_file[0]
model = args.model[0]
gpu = args.gpu
debug = args.debug
try:
    start_frame = args.start_frame
    end_frame = args.end_frame
except TypeError:
    print("No frame bounds provided, segmenting full movie")
    start_frame = 0
    end_frame = -1
print(f"Input file: {input_file}")
print(f"Output file: {output_file}")
print(f"Model: {model}")
print(f"GPU: {gpu}")
print(f"Start frame: {start_frame}")
print(f"End frame: {end_frame}")

# Load the model
print(f"Loading model {model}")
model = models.CellposeModel(gpu=gpu, pretrained_model=model)
# Set channels to [greyscale, no nuclei] or [0,0]
print("Setting channels to [0,0] (cytoplasm, no nuclei)")
chan = [0, 0]

# Read the input file
print(f"Reading input file {input_file}")
with ND2Reader(input_file) as images:
    images = list(images)
    used_images = images[start_frame:end_frame]
    print(
        f"Running segmentation on {len(images[start_frame:end_frame])} frames"
    )
    # Run segmentation on all frames
    masks = []
    flows = []
    probs = []
    # Run on single core or GPU if available
    for image in tqdm(
        used_images,
        desc="Frames",
        unit="frame",
    ):
        # Speed up with tile=False, uses more memory
        mask, flow, _ = model.eval(
            image, diameter=None, channels=chan, tile=False, flow_threshold=0.8
        )
        flows.append(flow[0])
        probs.append(flow[2])
        masks.append(mask)

    # Stack the masks
    masks = np.stack(masks)
    # Save the mask
    print(f"Saving to {output_file}")
    # Coerce to single channel
    masks = np.clip(masks, 0, 1)
    # Manually save the masks
    tifffile.imwrite(output_file, masks)
    if debug:
        print("Saving debug files")
        flows = np.stack(flows)
        # Save the flows
        flow_file = output_file + ".flow.tif"
        print(f"Saving flows to {flow_file}")
        tifffile.imwrite(flow_file, flows, bigtiff=True)
        # Save the probs
        prob_file = output_file + ".prob.tif"
        print(f"Saving probs to {prob_file}")
        tifffile.imwrite(prob_file, probs, bigtiff=True)


#
# run_segmentation.py ends here
