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

    def __init__(self, root_dir, pre_cache=False):
        self.root_dir = root_dir
        self.file_list = glob.glob(f"{root_dir}/*.tif")
        self.masks = [f.replace('.tif', '_mask.tif') for f in self.file_list]
        self.length = -1 # Initialize to a fake length
        # How do we keep track of our data?
        # Consider: namedtuple (fancy tuple)
        self.fileTuple = namedtuple('fileTuple', ['filename', 'num_frames', 'max_index'])
        self.fileTuples = []
        self.frameTuple = namedtuple('frameTuple', ['filename', 'frame', 'num_cells', 'max_index'])
        self.frameTuples = []
        # I need to pull out the file name, num_frames, and max_index
        # then for each frame in each file I neeed to collect its associated file name,
        # frame num, num_cells, and max_index (of cells)
<<<<<<< Updated upstream
        cellnumidx =-1 # Initialize prior to count for total cell index
        for iFile in self.file_list:
            filename = iFile
            framelen = -1
            cellnum =-1 # Initialize prior to count for total cell in each frame
            with tifffile.TiffFile(iFile) as tif:
                for iFrame in tif.pages:
                    num_labels, _ = cv2.connectedComponents(iFrame)
                    cellnumidx += num_labels
                    cellnum += num_labels
                    framelen += 1
=======
        cellnumidx = 0  # Initialize prior to count for total cell index
        for i, iFile in tqdm(
            enumerate(self.file_list),
            desc="Reading TIF Files",
            total=len(self.file_list),
        ):
            filename = iFile
            framelen = 0
            cellnum = 0  # Initialize prior to count for total cell in each frame
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
>>>>>>> Stashed changes
                # Now create the named tuple instance with the calculated values
                frame_info = self.frameTuple(filename=filename, frame=framelen, num_cells=cellnum, max_index=cellnumidx)
                self.frameTuples.append(frame_info)
            # Now create the named tuple instance with the calculated values
            file_info = self.fileTuple(filename=filename, num_frames=framelen, max_index=cellnumidx)
            self.fileTuples.append(file_info)
<<<<<<< Updated upstream
=======
        # Order frame and file tuples by max_index
        self.frameTuples = sorted(self.frameTuples, key=lambda x: x.max_index)
        self.fileTuples = sorted(self.fileTuples, key=lambda x: x.max_index)
        # Create a cache dictionary for items that we've already loaded
        self.cache = {}
        if pre_cache == True:
            print("Pre-caching dataset")
            for i in tqdm(range(len(self)), desc="Precaching", unit="sample"):
                self[i]
>>>>>>> Stashed changes

    def __len__(self):
        if self.length == -1:
            self.length = 0
            for iFile in self.file_list:
<<<<<<< Updated upstream
                with tifffile.TiffFile(iFile) as tif:
                    for iFrame in tif.pages:
                        num_labels, _ = cv2.connectedComponents(iFrame)
                        self.length += num_labels
=======
                tif = io.imread(iFile)
                for iFrame in tif:
                    _, _, _, centroids = cv2.connectedComponentsWithStats(
                        iFrame, 1, cv2.CV_32S
                    )
                    num_labels = len(centroids)
                    self.length += num_labels
>>>>>>> Stashed changes
        return self.length
    
    def findFile(self, idx):
        left, right = 0, len(self.fileTuples) - 1
        while left <= right:
            mid = left + (right - left) // 2  # Calculate the middle index
            # Check if the idx is present at mid
<<<<<<< Updated upstream
            if self.fileTuples[mid].max_index <= idx and self.fileTuples[mid-1].max_index > idx:
=======
            if (
                self.fileTuples[mid].max_index <= idx
                and self.fileTuples[mid - 1].max_index > idx
            ):
>>>>>>> Stashed changes
                return self.fileTuples[mid].filename  # Return the filename if found
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
            if self.frameTuples[mid].max_index <= idx and self.frameTuples[mid-1].max_index > idx:
                return self.frameTuples[mid].frame  # Return the frame if found
            # If idx is greater, ignore the left half
            elif self.frameTuples[mid].max_index < idx:
                left = mid + 1
            # If idx is smaller, ignore the right half
            else:
                right = mid - 1

        return None  # Return None if the idx is not found

<<<<<<< Updated upstream
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
=======
    def findFile(self, idx):
        """Naive search for file index"""
        num_files = len(self.fileTuples)
        for i in range(num_files):
            if idx <= self.fileTuples[i].max_index:
                return self.fileTuples[i - 1].filename
        raise ValueError("File index not found")

    def findFrame(self, idx):
        """Naive search for file index"""
        for i, frame in enumerate(self.frameTuples):
            if idx <= frame.max_index:
                return i - 1
        raise ValueError("Frame index not found")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Check if we have already loaded this item
        if idx in self.cache:
            return self.cache[idx]
        # If not, we need to load it
        centroidTuple = self.centroidTuples[idx]
        if centroidTuple.cellidx != idx:
            raise ValueError("Cell index mismatch, this should not happen")
>>>>>>> Stashed changes

        # We are looking for where a certain cell idx is. 
        # We need to use the named tuples that have built to find the file and frame location of said cell.
        filename = self.findFile(self, idx)
        frame = self.findFrame(self, idx)
        # Then, take the frame, find its location in the frame
        mask = tifffile.TiffFile(filename)
        mask_frame = mask.pages(frame)
        cell_idx = -(idx - self.frameTuples[frame].max_index)
        numlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_frame, 1, cv2.CV_32S)
        # Parse filename pull out respective.nd2 frame
        parts = filename.split('_')
        date_part = parts[0]
        movie = glob.glob(f"{self.root_dir}/{date_part}*.nd2")
        movie = ND2Reader(movie)
<<<<<<< Updated upstream
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
        cell_class = parts[1] #examples: WT, cyto, csome, pscome (as GFP localizations)
        # Return the sample + the class as a dictionary. sample = {'image': image (512x512xN), 'class': class}
        sample = {'image': cropped_movie, 'mask': cropped_mask, 'class': cell_class}

=======
        # Get every channel manually from this janky library yo
        num_channels = movie.sizes["c"]
        first_channel = movie.get_frame(frame)
        movie_frame = np.zeros(
            (first_channel.shape[0], first_channel.shape[1], num_channels + 1)
        )
        for c in range(num_channels):
            frame_2d = movie.get_frame_2D(t=frame, c=c)
            movie_frame[:, :, c] = frame_2d
        # Then, crop 512x512 around our cell of interest in the ND2 and mask (will need to connect the mask back to its associated Nd2)
        # Define the region to crop based on specified location and size
        x = centroidTuple.centroid[0]
        y = centroidTuple.centroid[1]
        x_min = 0
        x_max = mask_frame.shape[1]
        y_min = 0
        y_max = mask_frame.shape[0]
        if x < x_min + 256:
            x1 = x_min
            x2 = x_min + 512
        elif x > x_max - 256:
            x1 = x_max - 512
            x2 = x_max
        else:
            x1 = int(x - 256)
            x2 = int(x + 256)
        if y < y_min + 256:
            y1 = y_min
            y2 = y_min + 512
        elif y > y_max - 256:
            y1 = y_max - 512
            y2 = y_max
        else:
            y1 = int(x - 256)
            y2 = int(x + 256)
        # Crop the images
        cropped_mask = mask_frame[y1:y2, x1:x2]
        cropped_movie = movie_frame[y1:y2, x1:x2]
        # If the cropped images are too small, augment
        if cropped_mask.shape[0] < 512 or cropped_mask.shape[1] < 512:
            # Pad the images
            cropped_mask = np.pad(
                cropped_mask,
                (
                    (0, 512 - cropped_mask.shape[0]),
                    (0, 512 - cropped_mask.shape[1]),
                ),
                "constant",
                constant_values=0,
            )
            cropped_movie = np.pad(  # Pad the movie with zeros as well
                cropped_movie,
                (
                    (0, 512 - cropped_movie.shape[0]),
                    (0, 512 - cropped_movie.shape[1]),
                    (0, 0),
                ),
                "constant",
                constant_values=0,
            )
        # We will need to maintain the classification ID based on what mask its taking the cell from
        cell_class = parts[
            1
        ]  # examples: WT, cyto, csome, pscome (as GFP localizations)
        # We will need to one-hot encode the class
        if cell_class == "WT":
            cell_class = 0
        elif cell_class == "cyto":
            cell_class = 1
        elif cell_class == "csome":
            cell_class = 2
        elif cell_class == "pcsome":
            cell_class = 3
        else:
            raise ValueError(f"Cell class {cell_class} not found")
        class_dict = {  # One-hot encoding
            0: [1, 0, 0, 0],
            1: [0, 1, 0, 0],
            2: [0, 0, 1, 0],
            3: [0, 0, 0, 1],
        }
        # Stack the image and the mask
        cropped_movie[:, :, -1] = cropped_mask
        # cropped_movie = np.stack([cropped_mask, cropped_movie], axis=0)
        # Return the sample + the class as a dictionary. sample = {'image': image (512x512xN), 'class': class}
        sample = {
            "image": torch.tensor(cropped_movie, dtype=torch.float32),
            "class": torch.tensor(class_dict[cell_class], dtype=torch.float32),
        }
        # Add the sample to the cache
        self.cache[idx] = sample
>>>>>>> Stashed changes
        return sample

# Load the data
<<<<<<< Updated upstream
data = ND2DataSet(root_dir='D:\ZachML\CellType')  # Manually change to initialize the dataset
=======
data = ND2DataSet(
    # root_dir="/Volumes/Extreme SSD/ZachML/CellType"
    root_dir="E:\ZachML\CellType"
)  # Manually change to initialize the dataset
>>>>>>> Stashed changes

# Print the length of the dataset
print(len(data))

# Load into minibatches of 128
train_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

# Iterating over the dataloader
<<<<<<< Updated upstream
for batch in train_loader[0]:
    print(batch)
=======
# for batch in train_loader:
#    print(batch["image"].shape)
#    print(batch["class"].shape)
>>>>>>> Stashed changes

# TODO: Define the model
# We'll train a per-channel model + a fusion model for when we have all channels.

class ConvNetClassifier(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(ConvNetClassifier, self).__init__()
        assert 1 <= num_channels <= 6, "num_channels must be between 1 and 6"

<<<<<<< Updated upstream
        # Define the layers for each channel
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1) for _ in range(num_channels)])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Define the fusion layer
        self.fusion = nn.Linear(16 * num_channels, 32)

        # Define the remaining layers
        self.fc1 = nn.Linear(..., ...)
        self.dropout = nn.Dropout(p=0.25) # To prevent overfittng
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
        x = F.softmax(x, dim=1) #To get class probabilities   

        return x

=======
        # Convolute that shit baybee
        self.convolutions = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
        )

        # Define the remaining layers
        self.linear = nn.Sequential(
            nn.Linear(24576, 512),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(512, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.convolutions(x.view((32, 6, 512, 512)))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


# Set up the training loop
# First construct a model to optimize
model = ConvNetClassifier(num_channels=6, num_classes=4)
# First we need to define our optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Second we need a loss function. This is classification, so we'll use cross entropy loss
criterion = nn.CrossEntropyLoss()

# Next, we figure out the device that we are training on
if torch.cuda.is_available():
    print("CUDA is available (GPU)")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("MPS is available (M1 Mac)")
    device = torch.device("mps")
else:
    print("GPU is not available (CPU) :(")
    device = torch.device("cpu")

# Move the model to the device
model.to(device)

# Now we can train the model
num_epochs = 5
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}")
    for i, batch in tqdm(
        enumerate(train_loader), desc="Batches", unit="batch", total=len(train_loader)
    ):
        # Get the inputs and labels
        inputs = batch["image"].to(device)
        labels = batch["class"].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()  # Calculate the gradients
        optimizer.step()  # Update the weights

        # Print statistics
        #if i % 100 == 0:
        print(f"Batch {i+1}: loss = {loss.item():.3f}")

>>>>>>> Stashed changes
# train_classification_model.py ends here
