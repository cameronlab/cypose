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
import os
from tqdm import tqdm


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
        if os.path.isdir(root_dir) == False:
            raise ValueError(f"Root directory {root_dir} does not exist")
        self.file_list = glob.glob(f"{root_dir}/*.tiff")
        if len(self.file_list) == 0:
            raise ValueError("No .tif files found in root directory")
        self.masks = [f.replace(".tiff", "_mask.tiff") for f in self.file_list]
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
        self.centroidTuple = namedtuple(
            "centroidTuple", ["fileidx", "frameidx", "cellidx", "centroid"]
        )
        self.centroidTuples = []
        # I need to pull out the file name, num_frames, and max_index
        # then for each frame in each file I neeed to collect its associated file name,
        # frame num, num_cells, and max_index (of cells)
        cellnumidx = 0  # Initialize prior to count for total cell index
        for i, iFile in tqdm(enumerate(self.file_list), desc="Reading TIF Files", total=len(self.file_list)):
            filename = iFile
            framelen = 0
            cellnum = (
                0
            )  # Initialize prior to count for total cell in each frame
            tif = io.imread(iFile)
            for j, iFrame in enumerate(tif):
                _, _, _, centroids = cv2.connectedComponentsWithStats(
                    iFrame, 1, cv2.CV_32S
                )                
                for centroid in centroids:
                    centroid_info = self.centroidTuple(
                        fileidx=i,
                        frameidx=j,
                        cellidx=cellnumidx,
                        centroid=centroid,
                    )
                    self.centroidTuples.append(centroid_info)
                    cellnumidx += 1
                num_labels = len(centroids)
                cellnum += num_labels
                # Now create the named tuple instance with the calculated values
                frame_info = self.frameTuple(
                    filename=filename,
                    frame=framelen,
                    num_cells=cellnum,
                    max_index=cellnumidx,
                )
                framelen += 1
                self.frameTuples.append(frame_info)
                # Now create the named tuple instance with the calculated values
            file_info = self.fileTuple(
                filename=filename, num_frames=framelen, max_index=cellnumidx
            )
            self.fileTuples.append(file_info)
        # Order frame and file tuples by max_index
        self.frameTuples = sorted(self.frameTuples, key=lambda x: x.max_index)
        self.fileTuples = sorted(self.fileTuples, key=lambda x: x.max_index)

    def __len__(self):
        if self.length == -1:
            self.length = 0
            for iFile in self.file_list:
                tif = io.imread(iFile)
                for iFrame in tif:
                    _, _, _, centroids = cv2.connectedComponentsWithStats(
                        iFrame, 1, cv2.CV_32S
                    )                
                    num_labels = len(centroids)
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

    def findFile(self, idx):
        """Naive search for file index"""
        num_files = len(self.fileTuples)
        for i in range(num_files):
            if idx <= self.fileTuples[i].max_index:
                return self.fileTuples[i-1].filename
        raise ValueError("File index not found")

    def findFrame(self, idx):
        """Naive search for file index"""
        for i, frame in enumerate(self.frameTuples):
            if idx <= frame.max_index:
                return i-1
        raise ValueError("Frame index not found")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        centroidTuple = self.centroidTuples[idx]
        if centroidTuple.cellidx != idx:
            raise ValueError("Cell index mismatch, this should not happen")

        # We are looking for where a certain cell idx is.
        # We need to use the named tuples that have built to find the file and frame location of said cell.
        filename = self.file_list[centroidTuple.fileidx]
        frame = centroidTuple.frameidx
        # filename = self.findFile(idx)
        # frame = self.findFrame(idx)
        # Then, take the frame, find its location in the frame
        mask = io.imread(filename)
        mask_frame = mask[frame]
        # Parse filename pull out respective.nd2 frame
        parts = filename.split("_")
        date_part = parts[0]
        movie = glob.glob(f"{date_part}*.nd2")[0]
        movie = ND2Reader(movie)
        movie_frame = movie[frame]
        # Then, crop 512x512 around our cell of interest in the ND2 and mask (will need to connect the mask back to its associated Nd2)
        # Define the region to crop based on specified location and size
        x = centroidTuple.centroid[0]
        y = centroidTuple.centroid[1]
        x_min = 0
        x_max = mask_frame.shape[1]
        y_min = 0
        y_max = mask_frame.shape[0]
        if x < x_min + 256:
            x1 = 0
            x2 = 512
        elif x > x_max - 256:
            x1 = 512 - 512
            x2 = 512
        else:
            x1 = int(x - 256)
            x2 = int(x + 256)
        if y < y_min + 256:
            y1 = 0
            y2 = 512
        elif y > y_max - 256:
            y1 = 512 - 512
            y2 = 512
        else:
            y1 = int(x - 256)
            y2 = int(x + 256)
        # Crop the images
        cropped_mask = mask_frame[y1:y2,x1:x2]
        cropped_movie = movie_frame[y1:y2,x1:x2]
        # We will need to maintain the classification ID based on what mask its taking the cell from
        cell_class = parts[
            1
        ]  # examples: WT, cyto, csome, pscome (as GFP localizations)
        # Return the sample + the class as a dictionary. sample = {'image': image (512x512xN), 'class': class}
        sample = {
            "image": cropped_movie.astype(np.float32),
            "mask": cropped_mask.astype(np.uint8),
            #"class": cell_class.astype(np.uint8),
        }

        return sample


# Load the data
data = ND2DataSet(
    root_dir="/Volumes/Extreme SSD/ZachML/CellType"
    #root_dir="D:\ZachML\CellType"
)  # Manually change to initialize the dataset

# Print the length of the dataset
# print(len(data))

# Load into minibatches of 128
train_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

# Iterating over the dataloader
for batch in train_loader:
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
