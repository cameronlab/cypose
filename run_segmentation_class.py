import argparse
import os
import numpy as np
import torch
import tifffile
from tqdm import tqdm
from skimage import io
from nd2reader import ND2Reader
from cellpose import models, denoise


class SegmentationRunner:
    def __init__(self, args):
        self.input_file = args.input_file[0]
        self.output_file = args.output_file[0]
        self.model_name = args.model[0]
        self.gpu = args.gpu
        self.denoise_p = args.denoise
        self.niter = args.niter if args.niter else None
        self.flow_threshold = args.flow_threshold if args.flow_threshold else 0.75
        self.debug = args.debug
        self.size = args.size
        self.start_frame = args.start_frame if args.start_frame is not None else 0
        self.end_frame = args.end_frame if args.end_frame is not None else -1
        self.device = self.get_device()
        self.file_type = self.get_file_type()
        self.model, self.chan = self.load_model()
        self.size_model = self.load_size_model()
        self.denoise_model = self.load_denoise_model() if self.denoise_p else None
        self.masks, self.flows, self.probs = [], [], []


    def get_device(self):
        print("Checking for CUDA")
        if torch.cuda.is_available():
            print("CUDA is available (GPU)")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("MPS is available (M1 Mac)")
            return torch.device("mps")
        print("CUDA is not available (CPU)")
        return torch.device("cpu")

    def get_file_type(self):
        if self.input_file.endswith(".nd2"):
            return "nd2"
        elif self.input_file.endswith(".tif") or self.input_file.endswith(".tiff"):
            return "tif"
        raise ValueError("Unsupported file type")

    def load_model(self):
        if os.path.exists(self.model_name):
            if "2ch" in self.model_name:
                print(f"Using 2 channel model {self.model_name}")
                chan = [0, 1]
            else:
                print(f"Using 1 channel model {self.model_name}")
                chan = [0, 0]
            print(f"Loading model {self.model_name}")
            return models.CellposeModel(gpu=self.gpu, pretrained_model=self.model_name, device=self.device), chan
        print(f"Using base model {self.model_name}")
        return models.CellposeModel(gpu=self.gpu, model_type=self.model_name, device=self.device), [0, 0]

    def load_size_model(self):
        print("Loading size estimation model")
        model_type = "cyto2"
        pretrained_size = models.size_model_path(model_type)
        size_model = models.SizeModel(device=self.device, pretrained_size=pretrained_size, cp_model=self.model)
        size_model.model_type = model_type
        return size_model

    def load_denoise_model(self):
        print("Loading denoising model")
        return denoise.DenoiseModel(device=self.device, model_type="denoise_cyto3")

    def estimate_size(self):
        if self.size is None:
            if self.file_type == "nd2":
                with ND2Reader(self.input_file) as images:
                    first_frame = self.get_movie_frame(images, self.start_frame)
            else:
                images = tifffile.imread(self.input_file)
                first_frame = self.get_movie_frame(images, self.start_frame)[0, :, :]
            self.size = self.size_model.eval(first_frame, channels=self.chan)[0]
            print(f"Size estimated as {self.size} from first frame")

    def get_movie_frame(self, movie, frame_idx):
        if hasattr(movie, "bundle_axes"):
            movie.bundle_axes = ["y", "x"]
            return np.array(movie.get_frame(frame_idx), dtype=np.uint16)
        return np.array(movie[frame_idx], dtype=np.uint16)

    def run_segmentation(self):
        print(f"Reading input file {self.input_file}")
        if self.file_type == "nd2":
            self.process_nd2()
        elif self.file_type == "tif":
            self.process_tif()
        self.save_output()

    def process_nd2(self):
        with ND2Reader(self.input_file) as images:
            self.end_frame = len(images) if self.end_frame == -1 else self.end_frame
            print(f"Running segmentation on {self.end_frame - self.start_frame} frames")
            self.estimate_size()
            for i in tqdm(range(self.start_frame, self.end_frame), desc="Frames", unit="frame"):
                self.segment_frame(self.get_movie_frame(images, i), i)


    def process_tif(self):
        images = tifffile.imread(self.input_file)
        print(images.shape)
        self.end_frame = len(images) if self.end_frame == -1 else self.end_frame
        print(f"Running segmentation on {self.end_frame - self.start_frame} frames")
        self.estimate_size()
        for i in tqdm(range(self.start_frame, self.end_frame), desc="Frames", unit="frame"):
            self.segment_frame(self.get_movie_frame(images, i)[0, :, :], i)


    def segment_frame(self, image, frame_idx):
        if self.denoise_p:
            image = self.denoise_model.eval(image, channels=self.chan)
        mask, flow, _ = self.model.eval(image, diameter=self.size, channels=self.chan, niter=self.niter,
                                        flow_threshold=self.flow_threshold)
        self.masks.append(mask)
        self.flows.append(flow[0])
        self.probs.append(flow[2])

    def save_output(self):
        print(f"Saving to {self.output_file}")
        tifffile.imwrite(self.output_file, np.stack(self.masks))
        if self.debug:
            print("Saving debug files")
            tifffile.imwrite(self.output_file + ".flow.tif", np.stack(self.flows), bigtiff=True)
            tifffile.imwrite(self.output_file + ".prob.tif", np.stack(self.probs), bigtiff=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run segmentation on a movie")
    parser.add_argument("--input_file", type=str, nargs=1, required=True, help="Input file")
    parser.add_argument("--output_file", type=str, nargs=1, required=True, help="Output file")
    parser.add_argument("--model", type=str, nargs=1, required=True, help="Model to segment")
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction)
    parser.add_argument("--start_frame", type=int,)
    parser.add_argument("--end_frame", type=int)
    parser.add_argument("--denoise", action=argparse.BooleanOptionalAction)
    parser.add_argument("--size", type=int)
    parser.add_argument("--niter", type=int)
    parser.add_argument("--flow_threshold", type=float)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    segmentation_runner = SegmentationRunner(args)
    segmentation_runner.run_segmentation()
