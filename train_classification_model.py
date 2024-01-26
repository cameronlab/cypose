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
import cv2
import tifffile
from nd2reader import ND2Reader
import numpy as np
import glob as glob
from skimage import io
from collections import namedtuple
from tqdm import tqdm
import os
from functools import lru_cache
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

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
        # Order frame and file tuples by max_index
        self.frameTuples = sorted(self.frameTuples, key=lambda x: x.max_index)
        self.fileTuples = sorted(self.fileTuples, key=lambda x: x.max_index)
        # Create a cache dictionary for items that we've already loaded
        self.cache = {}
        if pre_cache:
            print("Pre-caching dataset")
            for i in tqdm(range(len(self)), desc="Precaching", unit="sample"):
                self[i]

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

        # We are looking for where a certain cell idx is.
        # We need to use the named tuples that have built to find the file and frame location of said cell.
        filename = self.findFile(self, idx)
        frame = self.findFrame(self, idx)
        # Then, take the frame, find its location in the frame
        mask = tifffile.TiffFile(filename)
        mask_frame = mask.pages(frame)
        numlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_frame, 1, cv2.CV_32S
        )
        # Parse filename pull out respective.nd2 frame
        parts = filename.split("_")
        date_part = parts[0]
        movie = glob.glob(f"{self.root_dir}/{date_part}*.nd2")
        movie = ND2Reader(movie)
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
        return sample


class FastND2DataSet(torch.utils.data.Dataset):
    """
    Faster implementation of the nd2 dataset loader.
    Key improvements:
        - Preload all masks per-file to reduce filesystem thrashing
        - Save processed mask objects to memory to speed up subsequent loads
    If needed we will also implement an async buffer class over top for when there are too many files to fit in memory.
    Future improvements:
        - Write each frame to its own file on disk for our buffer class when dealing with larger than menmory datasets
    """

    def __init__(self, root_dir, cache_path=None, calc_cache=False):
        self.root_dir = root_dir
        self.file_list = glob.glob(f"{root_dir}/*.tiff")
        self.tile_size = 64
        # Make sure we have files
        if len(self.file_list) == 0:
            raise ValueError(f"No files found in root directory {root_dir}")
        self.masks = [f.replace(".tiff", "_mask.tif") for f in self.file_list]
        cellnumidx = 0  # Initialize prior to count for total cell index
        if cache_path is None:
            self.cache_path = f"{root_dir}/cache"
            if not os.path.exists(self.cache_path):
                os.mkdir(self.cache_path)
        # Check if we have cached data
        if not calc_cache:
            cached_files = glob.glob(f"{self.cache_path}/*.pth")
            self.length = len(cached_files)
            return
        # Manually start a tqdm progress bar
        pbar = tqdm(desc="Caching cells", unit="cell", total=35000)
        # Iterate over every tif file
        for i, iFile in enumerate(self.file_list):
            print(f"Loading file {i+1} of {len(self.file_list)}, {iFile}")
            filename = iFile
            tif = io.imread(iFile)
            movie, parts = self.getAssocND2(filename)
            for j, iFrame in enumerate(tif):
                _, _, _, centroids = cv2.connectedComponentsWithStats(
                    iFrame, 1, cv2.CV_32S
                )
                movie_frame = self.getMovieFrame(movie, j)
                for centroid in centroids:
                    # Process the mask
                    x, y = centroid[0], centroid[1]
                    cropped_mask, cropped_movie = self.padMask(
                        x, y, iFrame, movie_frame
                    )
                    # If our mask is empty, skip it
                    if cropped_mask.sum() < 10:
                        #print(f"Skipping empty mask at {x},{y}")
                        continue
                    cell_class = self.oneHotEncode(parts)
                    stacked_movie = self.stackMovieAndMask(cropped_mask, cropped_movie)
                    # Add the sample to the cache
                    sample = {
                        "image": torch.tensor(stacked_movie, dtype=torch.float32),
                        "class": torch.tensor(cell_class, dtype=torch.float32),
                    }
                    cellnumidx += 1
                    # Save the sample to disk
                    torch.save(sample, f"{self.cache_path}/{cellnumidx}_data.pth")
                    # Save a thumbnail too lol
                    # self.generateThumbnails(stacked_movie, cellnumidx)
                    # Update the progress bar
                    pbar.update(1)
        # Calculate the length of the dataset
        cached_files = glob.glob(f"{self.cache_path}/*.pth")
        self.length = len(cached_files)

    def __len__(self):
        return self.length

    @lru_cache(maxsize=1)
    def getAssocND2(self, filename):
        """
        Given a tif filename, load the associated nd2 file
        """
        parts = os.path.basename(filename).split("_")
        date_part = parts[0]
        movie_path = glob.glob(f"{self.root_dir}/{date_part}*.nd2")[0]
        if len(movie_path) == 0:
            raise ValueError(f"No associated ND2 file found for {self.root_dir}/{date_part}*.nd2")
        movie = ND2Reader(movie_path)
        return movie, parts

    def getMovieFrame(self, movie, frame: int):
        """
        Given a movie and a frame, load the frame from the movie
        """
        # Get every channel manually because nd2reader is implemented poorly
        # num_channels = movie.sizes["c"]
        # first_channel = movie.get_frame(frame)
        # movie_frame = np.zeros(
        #     (first_channel.shape[0], first_channel.shape[1], num_channels + 1)
        # )
        # for c in range(num_channels):
        #     frame_2d = movie.get_frame_2D(t=frame, c=c)
        #     movie_frame[:, :, c] = frame_2d
        movie.bundle_axes = ['y', 'x', 'c']
        movie_frame = movie.get_frame(frame)
        frame_array = np.zeros((movie_frame.shape[0], movie_frame.shape[1], 6), dtype=np.float32)
        frame_array[:,:,0:5] = movie_frame
        return frame_array

    def padMask(self, x, y, mask_frame, movie_frame):
        """Pad the mask to the correct size"""
        # Make sure our size is even
        size = self.tile_size
        if size % 2 != 0:
            raise ValueError("Data size must be even for automated padding")
        # Calculate sizes
        half_size = size // 2
        movie_x, movie_y = movie_frame.shape[1], movie_frame.shape[0]
        mask_x, mask_y = mask_frame.shape[1], mask_frame.shape[0]
        # Make sure the mask is the same size as the movie
        if movie_x != mask_x or movie_y != mask_y:
            raise ValueError("Mask and movie must be the same size")
        x_min, x_max = 0, mask_x
        y_min, y_max = 0, mask_y
        x1 = int(max(x - half_size, x_min))
        x2 = int(min(x + half_size, x_max))
        y1 = int(max(y - half_size, y_min))
        y2 = int(min(y + half_size, y_max))
        # Crop the images
        cropped_mask = mask_frame[y1:y2, x1:x2]
        cropped_movie = movie_frame[y1:y2, x1:x2]
        # If the cropped images are too small, augment
        mask_shape = cropped_mask.shape
        if mask_shape[0] < size or mask_shape[1] < size:
            # Pad the images
            cropped_mask = np.pad(
                cropped_mask,
                ((0, size - mask_shape[0]), (0, size - mask_shape[1])),
                "constant",
                constant_values=0,
            )
            cropped_movie = np.pad(
                cropped_movie,
                ((0, size - mask_shape[0]), (0, size - mask_shape[1]), (0, 0)),
                "constant",
                constant_values=0,
            )
        return cropped_mask, cropped_movie

    def oneHotEncode(self, parts):
        # We will need to maintain the classification ID based on what mask its taking the cell from
        cell_class = parts[1]  
        class_dict = {  # One-hot encoding
            "WT": [1, 0, 0, 0],
            "cyto": [0, 1, 0, 0],
            "csome": [0, 0, 1, 0],
            "pcsome": [0, 0, 0, 1],
        }
        return class_dict[cell_class]

    def stackMovieAndMask(self, cropped_mask, cropped_movie):
        """Stack the movie and mask"""
        cropped_movie[:, :, -1] = cropped_mask
        if cropped_movie.shape != (self.tile_size,self.tile_size,6):
            raise ValueError(f"Movie shape is incorrect. Actual is {cropped_movie.shape}, expected is ({self.tile_size},{self.tile_size},6)")
        return cropped_movie

    # Iterate through dataloader and generate thumbnails for each movie
    def generateThumbnails(self, stacked_movie, cell_num):
        """
        Take a batch of images and generate a tiled 3x2 thumbnail of the 6 channels for each image
        """
        # Make an empty image 🥺
        tiled_image = np.zeros((self.tile_size * 3, self.tile_size * 2))
        # Iterate over each channel
        for channel_num in range(6):
            # Get the image
            channel_image = stacked_movie[:, :, channel_num]
            # Calculate the x and y coordinates
            x = channel_num % 2
            y = channel_num // 2
            # Calculate the x and y coordinates
            x1 = x * self.tile_size
            x2 = x1 + self.tile_size
            y1 = y * self.tile_size
            y2 = y1 + self.tile_size
            # Insert the image into the tiled image
            tiled_image[y1:y2, x1:x2] = channel_image
        # Save the tiled image
        tiled_image = tiled_image.astype(np.uint16)
        io.imsave(f"{self.root_dir}/thumbnails/thumb_c{cell_num}.tif", tiled_image)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Ooops, off by one
        idx += 1
        # Check if we have already loaded this item
        try:
            data = torch.load(f"{self.cache_path}/{idx}_data.pth")
            return data
        except KeyError:
            raise ValueError("Index not found in cache, someone done goofed.")



# Load the data
data = FastND2DataSet(
    root_dir="/Volumes/Extreme SSD/ZachML/CellType"
    #root_dir="E:\ZachML\CellType"
)  # Manually change to initialize the dataset

# Print the length of the dataset
# print(len(data))

# Define the batch size
batch_size = 32
# Split the data into train and test sets
train_dataset, test_dataset = torch.utils.data.random_split(
    data, [0.8, 0.2]
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class ConvNetClassifier(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(ConvNetClassifier, self).__init__()
        assert 1 <= num_channels <= 6, "num_channels must be between 1 and 6"

        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8 * 4, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Can we learn anything from our data at all?
# Use an autoencoder to find out
class ConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 6, kernel_size=2, stride=2),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.permute(0, 2, 3, 1)
        return x

# Set up the training loop
# First construct a model to optimize
model = ConvNetClassifier(num_channels=6, num_classes=4)
# model = ConvolutionalAutoencoder()
# First we need to define our optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# Second we need a loss function. This is classification, so we'll use cross entropy loss
criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()

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

# Set up tensorboard
writer = SummaryWriter()

# Define an accuracy function
def model_accuracy(outputs, labels):
    """
    Calcluate per-class accuracy.
    """
    return torch.eq(torch.argmax(outputs, dim=1), torch.argmax(labels, dim=1)).sum() / len(labels)


# Now we can train the model
num_epochs = 30
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}")
    for i, batch in tqdm(
        enumerate(train_loader), desc="Training", unit="batch", total=len(train_loader)
    ):
        # Get the inputs and labels
        inputs = batch["image"].to(device)
        labels = batch["class"].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        # Calculate the loss
        train_loss = criterion(outputs, labels)
        # loss = criterion(outputs, inputs)
        # Calculate the accuracy
        train_accuracy = model_accuracy(outputs, labels)

        # Backward pass
        train_loss.backward()  # Calculate the gradients
        optimizer.step()  # Update the weights

        # Print statistics
        # if i % 100 == 0:
        #print(f"Batch {i+1}: loss = {loss.item():.3f}")
        writer.add_scalar("Loss/train", train_loss.item(), epoch * len(train_loader) + i)
        writer.add_scalar("Accuracy/train", train_accuracy.item(), epoch * len(train_loader) + i)

    # Evaluate the model
    # losses = torch.tensor((len(test_loader))).to(device)
    # accuracies = torch.tensor((len(test_loader))).to(device)
    losses = []
    accuracies = []
    for i, batch in tqdm(
        enumerate(test_loader), desc="Testing", unit="batch", total=len(test_loader)
    ):
        # Get the inputs and labels
        inputs = batch["image"].to(device)
        labels = batch["class"].to(device)

        with torch.no_grad():
            outputs = model(inputs)
            test_loss = criterion(outputs, labels)
            test_accuracy = model_accuracy(outputs, labels)

        # losses[i] = test_loss
        # accuracies[i] = test_accuracy.item()
        losses.append(test_loss)
        accuracies.append(test_accuracy.item())
    
    # test_loss = losses.mean()
    # test_accuracy = accuracies.mean()
    test_loss = torch.tensor(losses).mean()
    test_accuracy = torch.tensor(accuracies).mean()

    writer.add_scalar("Loss/test", test_loss.item(), epoch * len(train_loader))
    writer.add_scalar("Accuracy/test", test_accuracy.item(), epoch * len(train_loader))

# train_classification_model.py ends here
