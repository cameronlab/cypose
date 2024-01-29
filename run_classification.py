# Run a pretrained classification model on a given folder

import argparse
import torch
from glob import glob
from train_classification_model import ConvNetClassifier, ND2DataSet
import os
from collections import namedtuple
from skimage import io
import cv2
from nd2reader import ND2Reader
import numpy as np
from tqdm import tqdm
import torch

if torch.cuda.is_available():
    print("CUDA is available (GPU)")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("MPS is available (M1 Mac)")
    device = torch.device("mps")
else:
    print("GPU is not available (CPU) :(")
    device = torch.device("cpu")


def load_model(model_path, model_arch=ConvNetClassifier, device=device):
    # Initialize the model
    model = model_arch(num_channels=5, num_classes=4)
    # Load the model per-state dict
    saved_model = torch.load(model_path)
    model.load_state_dict(saved_model)
    model.eval()
    return model.to(device)


movieTuple = namedtuple("movieTuple", ["nd2", "tif"])


def load_data(data_dir):
    # First get all the nd2 files
    nd2s = sorted(glob(os.path.join(data_dir, "*.nd2")))
    # Search for matching tif files by basename of nd2
    matches = []
    for nd2 in nd2s:
        nd2_base = os.path.basename(nd2)
        tif = glob(f"{data_dir}*{nd2_base[:-4]}*.tif")
        if len(tif) == 1:
            matches.append(movieTuple(nd2, tif[0]))
        else:
            print(f"Skipping {nd2_base} because it has {len(tif)} matches")
    return matches


def movie_cells(nd2_path, tif_path, tile_size=64):
    # First load the tif
    tif = io.imread(tif_path)
    nd2 = ND2Reader(nd2_path)
    for frame_num, tif_frame in enumerate(tif):
        tif_frame = tif_frame.astype(np.uint8)
        # Threshold the max value to 1
        tif_frame[tif_frame > 0] = 1
        _, _, _, centroids = cv2.connectedComponentsWithStats(
            tif_frame, 1, cv2.CV_32S
        )
        nd2_frame = ND2DataSet.getMovieFrame(nd2, frame_num)
        for centroid in centroids:
            x, y = centroid
            cropped_mask, cropped_movie = ND2DataSet.padMask(
                x, y, tif_frame, nd2_frame, tile_size
            )
            # if cropped_mask.sum() < 10:
            # print(f"Skipping empty mask at {x},{y} in {tif}")
            # continue
            stacked_movie = ND2DataSet.stackMovieAndMask(cropped_mask, cropped_movie)
            sample = torch.tensor(stacked_movie, dtype=torch.float32)
            sample = sample.reshape(1, *sample.shape)
            yield sample.to(device), centroid, frame_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the classifier on a pytorch model directory"
    )
    parser.add_argument(
        "--model_path", type=str, help="Path to the model file you want to use"
    )
    parser.add_argument("--data_dir", type=str, help="Path to paired nd2 and tif files")
    args = parser.parse_args()

    # Load the model
    model = load_model(args.model_path)

    # Load the data
    data = load_data(args.data_dir)

    # Iterate over our matches
    for nd2, tif in data:
        print(f"Running model on {nd2} and {tif}")
        # Open a csv to write to
        with open(f"{nd2[:-4]}_classes.csv", "w") as f:
            f.write("frame,x,y,class\n")
            # Run the model on each cell
            for sample, centroid, frame_num in tqdm(movie_cells(nd2, tif), desc="Classifying"):
                cell_class = model.predict(sample)
                class_str = ND2DataSet.oneHotDecode(cell_class)
                f.write(f"{frame_num},{centroid[0]},{centroid[1]},{class_str}\n")
                # predictions = model(sample)[0]
                # f.write(
                #     f"{centroid[0]},{centroid[1]},{predictions[0]},{predictions[1]},{predictions[2]},{predictions[3]}\n"
                # )
