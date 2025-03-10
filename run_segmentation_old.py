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
from skimage import io
import tifffile  # Tiff file writing
import argparse  # Command line arguments
from cellpose import models  # Cellpose
from cellpose import denoise  # Denoising
from tqdm import tqdm  # Progress bar
import numpy as np
import torch
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run segmentation on a movie")
# Standard flags
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
# Experimental flags
parser.add_argument(
    "--denoise",
    action=argparse.BooleanOptionalAction,
    help="Run denoising before segmentation",
)
parser.add_argument(
    "--niter",
    metavar="niter",
    type=int,
    nargs="?",
    help="The number of iterations to run segmentation per-frame",
)
parser.add_argument(
    "--flow_threshold",
    metavar="flow_threshold",
    type=float,
    nargs="?",
    help="The flow threshold to use for segmentation",
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
model_name = args.model[0]
gpu = args.gpu
denoise_p = args.denoise  # denoise is a package, so denote as a prefix _p
niter = args.niter
flow_threshold = args.flow_threshold
debug = args.debug

# Check for frame bounds
try:
    start_frame = args.start_frame
    end_frame = args.end_frame
except TypeError:
    print("No frame bounds provided, segmenting full movie")
    start_frame = 0
    end_frame = -1
if start_frame is None:
    start_frame = 0
if end_frame is None:
    end_frame = -1

# Check for CUDA
print("Checking for CUDA")
if torch.cuda.is_available():
    print("CUDA is available (GPU)")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("MPS is available (M1 Mac)")
    device = torch.device("mps")
else:
    print("CUDA is not available (CPU)")
    device = torch.device("cpu")

# Check for experimental flags
if denoise_p:
    print("Denoising enabled")
if niter:
    niter = int(niter)
    print(f"Using {niter} iterations")
else:
    niter = None
if flow_threshold:
    flow_threshold = float(flow_threshold)
    print(f"Using flow threshold {flow_threshold}")
else:
    flow_threshold = 0.75
if debug:
    print("Debug mode enabled")

# Check filename to determine TIFF or ND2
if input_file.endswith(".nd2"):
    file_type = "nd2"
elif input_file.endswith(".tif") or input_file.endswith(".tiff"):
    file_type = "tif"

# Print out the arguments
print(f"Input file: {input_file}")
print(f"File type: {file_type}")
print(f"Output file: {output_file}")
print(f"Model: {model_name}")
print(f"GPU: {gpu}")
print(f"Using device: {device}")
print(f"Start frame: {start_frame}")
print(f"End frame: {end_frame}")

if os.path.exists(model_name):
    # Check if model has 2 channels in string name
    if "2ch" in model_name:
        print(f"Using 2 channel model {model_name}")
        chan = [0, 1]
    else:
        print(f"Using 1 channel model {model_name}")
        chan = [0, 0]

    # Load the model
    print(f"Loading model {model_name}")
    model = models.CellposeModel(
        # MPS is M1 Mac Support
        gpu=gpu,
        pretrained_model=model_name,
        device=device,
    )
else:
    print(f"Using base model {model_name}")
    model = models.CellposeModel(
        # MPS is M1 Mac Support
        gpu=gpu,
        model_type=model_name,
        device=device,
    )
    chan = [0, 0]

# Load the size estimation model
print("Loading size estimation model")
model_type = "cyto2"
pretrained_size = models.size_model_path(model_type)
size_model = models.SizeModel(
    device=device, pretrained_size=pretrained_size, cp_model=model
)
size_model.model_type = model_type
# Load new cellpose 3 denoising model
if denoise_p:
    print("Loading denoising model")
    denoise_model = denoise.DenoiseModel(
        device=device, model_type="denoise_cyto3"
    )

# Select the file reader
if file_type == "nd2":
    reader = ND2Reader
elif file_type == "tif":
    reader = tifffile.imread

print(f"Reading input file {input_file}")
size = None
with reader(input_file) as images:
    used_images = images[start_frame:end_frame]
    print(
        f"Running segmentation on {len(images[start_frame:end_frame])+1} frames"
    )
    # Run segmentation on all frames
    masks = []
    flows = []
    probs = []
    # Run on single core or GPU if available
    # TODO Fix to use minibatches
    for i, image in tqdm(
        enumerate(used_images),
        desc="Frames",
        unit="frame",
        total=len(used_images),
    ):
        # pbar = tqdm(
        #     enumerate(used_images),
        #     desc="Frames",
        #     unit="frame",
        #     total=len(used_images),
        # )
        # for i in range(0, len(used_images), batch_size):
        image = np.array(image)
        print(image.shape)
        # Grab array
        # image = np.array(used_images[i : i + batch_size])
        # Convert axis 0 to a list
        # image = list(image[i, :, :] for i in range(image.shape[0]))
        # image = np.array(image)
        # Size estimation (do once)
        # if size is None:
        size, _ = size_model.eval(image, channels=chan)
        # print(f"Size estimated as {size}")
        # Denoising
        if denoise_p:
            image = denoise_model.eval(image, channels=chan)
        # Speed up with tile=False, uses more memory
        mask, flow, _ = model.eval(
            image,
            diameter=size,
            channels=chan,
            #tile=True,
            niter=niter,
            flow_threshold=flow_threshold,
        )
        # for mask_i, flow_i in zip(mask, flow):
        #     masks.append(mask_i)
        #     flows.append(flow_i[0])
        #     probs.append(flow_i[2])
        flows.append(flow[0])
        probs.append(flow[2])
        masks.append(mask)

    # Stack the masks
    masks = np.stack(masks)
    # Save the mask
    print(f"Saving to {output_file}")
    # Coerce to single channel
    # masks = np.clip(masks, 0, 1)
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
