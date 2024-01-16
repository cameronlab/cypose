# train_classification_model.py --- Train classifier
#
# Filename: train_classification_model.py
# Author: Zach Maas and Clair Huffine
# Created: Tue Dec 19 15:36:15 2023 (-0700)
#
#

# Commentary:
#
# This file contains code to train a basic convolutional classifier on
# segmented movies.
#

# Code:

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import tifffile
from nd2reader import ND2Reader
import pandas as pd
import numpy as np
import glob as glob
from skimage import io
from collections import namedtuple


# Define the data loader to load nd2 files
# This needs to take a file, return the connected components
# and the labels. Each connected component should be a separate
# training example, and the labels should be the label for that
# connected component.
class ND2DataSet(torch.utils.data.Dataset):
    """
    Hierarchy:
    - Folder of movies, each movie has a label
    - Movie + masks, each movie has many frames
    - Frame, each frame has many cells
    - Cells, each cell has 1+ channels
    Cells are our smallest unit of data, something like 512x512x3 (we can resize)
    How do we load this?
    - Folder, we can grab every .tif using python's built in glob module (glob.glob(f"{data_path}/*.tif"))
    - Movie + masks
      - We load the movies using nd2reader library
      - We load the masks using tifffile library, using only frames that we have masks for
    - Cells
      - We use cv2 connectedcomponents to identify each cell in a mask
      - Then, we use each cell in a mask to subset the movie
      - With that subset, we return our 512x512x5 matrix for each cell.
    Functions we need:
    - Init function
      - Get the root directory
      - Get a list of every filename for each TIF/ND2 (matched)
      - Import some metadata CSV telling us what channels we want for each movie + labels
      NOTE: we will train red and bf as the same channel in our model.
    - Len function
      - We define length based on masks
      - Mathematically: cells / frame * frames / mask * number of masks
      - Maybe we can precalculate the total then? Calculate it the first time and then cache.
      - Iterate over every cell in every frame
    - Getitem function
      - We need to associate an integer index value with each cell so we can pull data
        - Lookup name tuple for masks:
          Example: namedtuple('fileTuple', ['filename', 'num_frames', 'max_index'])
          Gives us our cell location in a frame
      - Take the frame from our lookup, crop to 512x512, and pull data from nd2 file to make vector
      - We should keep the mask data as input for the model as well so it knows where to actually look
      - Figure out the class, this is a dictionary associated with each .tif file
        Make sure that we use pytorch's one-hot-encoding to make the class numeric.
      - Return the sample + the class as a dictionary.
        sample = {'image': image (512x512xN), 'class': class}
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = glob.glob(f"{root_dir}/*.tif")
        self.masks = [f.replace(".tif", "_mask.tif") for f in self.file_list]
        self.length = -1  # Initialize to a fake length
        # How do we keep track of our data?
        # Consider: namedtuple (fancy tuple)
        self.fileTuple = namedtuple(
            "fileTuple", ["filename", "num_frames", "max_index"]
        )
        self.fileTuples = []
        self.frameTuple = namedtuple(
            "frameTuple", ["filename", "frame", "num_cells", "max_index"]
        )
        self.frameTuples = []
        # I need to pull out the file name, num_frames, and max_index
        # then for each frame in each file I neeed to collect its associated file name,
        # frame num, num_cells, and max_index (of cells)
        cellnumidx = -1  # Initialize prior to count for total cell index
        for iFile in self.file_list:
            filename = iFile
            framelen = -1
            cellnum = (
                -1
            )  # Initialize prior to count for total cell in each frame
            with tifffile.TiffFile(iFile) as tif:
                for iFrame in tif.pages:
                    num_labels, _ = cv2.connectedComponents(iFrame)
                    cellnumidx += num_labels
                    cellnum += num_labels
                    framelen += 1
                # Now create the named tuple instance with the calculated values
                frame_info = self.frameTuple(
                    filename=filename,
                    frame=framelen,
                    num_cells=cellnum,
                    max_index=cellnumidx,
                )
                self.frameTuples.append(frame_info)
            # Now create the named tuple instance with the calculated values
            file_info = self.fileTuple(
                filename=filename, num_frames=framelen, max_index=cellnumidx
            )
            self.fileTuples.append(file_info)

    def __len__(self):
        if self.length == -1:
            self.length = 0
            for iFile in self.file_list:
                with tifffile.TiffFile(iFile) as tif:
                    for iFrame in tif.pages:
                        num_labels, _ = cv2.connectedComponents(iFrame)
                        self.length += num_labels
        return self.length

    def findFile(self, idx):
        left, right = 0, len(self.fileTuples) - 1
        while left <= right:
            mid = left + (right - left) // 2  # Calculate the middle index
            # Check if the idx is present at mid
            if (
                self.fileTuples[mid].max_index <= idx
                and self.fileTuples[mid - 1].max_index > idx
            ):
                return self.fileTuples[
                    mid
                ].filename  # Return the filename if found
            # If idx is greater, ignore the left half
            elif self.fileTuples[mid].max_index < idx:
                left = mid + 1
            # If idx is smaller, ignore the right half
            else:
                right = mid - 1

    def findFrame(self, idx):
        left, right = 0, len(self.frameTuples) - 1
        while left <= right:
            mid = left + (right - left) // 2  # Calculate the middle index
            # Check if the idx is present at mid
            if (
                self.frameTuples[mid].max_index <= idx
                and self.frameTuples[mid - 1].max_index > idx
            ):
                return self.frameTuples[mid].frame  # Return the frame if found
            # If idx is greater, ignore the left half
            elif self.frameTuples[mid].max_index < idx:
                left = mid + 1
            # If idx is smaller, ignore the right half
            else:
                right = mid - 1

        return None  # Return None if the idx is not found

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # We are looking for where a certain cell idx is.
        # We need to use the named tuples that have built to find the file and frame location of said cell.
        filename = self.findFile(self, idx)
        frame = self.findFrame(self, idx)
        # Then, take the frame, find its location in the frame
        mask = tifffile.TiffFile(filename)
        mask_frame = mask.pages(frame)
        cell_idx = -(idx - self.frameTuples[frame].max_index)
        numlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_frame, 1, cv2.CV_32S
        )
        # Parse filename pull out respective.nd2 frame
        parts = filename.split("_")
        date_part = parts[0]
        movie = glob.glob(f"{self.root_dir}/{date_part}*.nd2")
        movie = ND2Reader(movie)
        movie_frame = movie.pages(frame)
        # Then, crop 512x512 around our cell of interest in the ND2 and mask (will need to connect the mask back to its associated Nd2)
        # Define the region to crop based on specified location and size
        x = centroids(labels[cell_idx], 0)
        y = centroids(labels[cell_idx], 1)
        box = (x + 256, y + 256, x - 256, y - 256)
        # Crop the images
        cropped_mask = mask_frame.crop(box)
        cropped_movie = movie_frame.crop(box)
        # We will need to maintain the classification ID based on what mask its taking the cell from
        cell_class = parts[
            1
        ]  # examples: WT, cyto, csome, pscome (as GFP localizations)
        # Return the sample + the class as a dictionary. sample = {'image': image (512x512xN), 'class': class}
        sample = {
            "image": cropped_movie,
            "mask": cropped_mask,
            "class": cell_class,
        }

        return sample


# Load the data
data = ND2DataSet(
    root_dir="D:\ZachML\CellType"
)  # Manually change to initialize the dataset

# Print the length of the dataset
print(len(data))

# Load into minibatches of 128
train_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

# Iterating over the dataloader
for batch in train_loader[0]:
    print(batch)

# TODO: Define the model
# We'll train a per-channel model + a fusion model for when we have all channels.


class ConvNetClassifier(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(ConvNetClassifier, self).__init__()
        assert 1 <= num_channels <= 4, "num_channels must be between 1 and 4"

        # Define the layers for each channel
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
                for _ in range(num_channels)
            ]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Define the fusion layer
        self.fusion = nn.Linear(16 * num_channels, 32)

        # Define the remaining layers
        self.fc1 = nn.Linear(..., ...)
        self.dropout = nn.Dropout(p=0.25)  # To prevent overfittng
        self.fc2 = nn.Linear(..., num_classes)

    def forward(self, x):
        # Process each channel
        xs = [self.pool(F.relu(conv(x))) for conv in self.convs]

        # Flatten the outputs and concatenate
        xs = [x.view(x.size(0), -1) for x in xs]
        x = torch.cat(xs, dim=1)

        # Pass through the fusion layer
        x = F.relu(self.fusion(x))

        # Pass through the remaining layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)  # To get class probabilities

        return x


# train_classification_model.py ends here
