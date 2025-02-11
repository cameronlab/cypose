# Cyanosegment - Cyanobacteria Optimized Deep Cell Segmentation

<p align="center">
	⚠️ The main documentation for this project can be found at  <a href="https://cameronlab.github.io/cypose/">GitHub Pages</a> ⚠️
</p>

## Description
This script is designed to run segmentation on a cyanobacteria movie using the [Cellpose](https://github.com/MouseLand/cellpose) model. It reads an ND2 file, applies the specified model for segmentation, and saves the results as a TIFF stack. Included in this repository is our recommended model for segmentation of *Synechococcus sp.* Strain PCC 7002, finetuned from the base cellpose model. This model is saved in the *models* folder.

## Installation Instructions
To use this script, you need to have Python installed along with the following packages: *nd2reader*, *tifffile*, *argparse*, *cellpose*, *tqdm*, and *numpy*. You can install these using pip:
```bash
pip install nd2reader tifffile argparse cellpose tqdm numpy
```
Alternatively (and likely easier on Windows), you can install these using conda:
```
conda install -c conda-forge nd2reader tifffile argparse cellpose tqdm numpy
```
In some circumstances, you may need to install *pytorch* manually. You can do this using pip or conda to get proper GPU acceleration. See the [Pytorch website](https://pytorch.org/get-started/locally/) for more information.
## Execution Instructions
To run the script, use the following command in your terminal:
```bash
python run_segmentation_backup.py --input_file input.nd2 --output_file output.tiff --model model_name [--gpu] [--start_frame frame_number] [--end_frame frame_number] [--debug]
```
Replace *input.nd2* with the path to your input ND2 file, *output.tiff* with the desired output TIFF file name, and *model_name* with the path of the model you want to use. The script will run segmentation on all frames (or a specified subset) of the input file and save the results as a TIFF stack.

Optional arguments:
- *--gpu*: Use GPU for segmentation if available. This gives a ~2-10x speedup depending on your hardware.
- *--start_frame frame_number*: The first frame to start segmentation on. Default is 0.
- *--end_frame frame_number*: The last frame to end segmentation on. If not provided, the script will segment all frames.
- *--debug*: Run in debug mode, which saves additional output files (flows and probabilities).

Example usage:
```bash
python run_segmentation_backup.py --input_file data/movie.nd2 --output_file results/segmented.tiff --model models/7002_CAH_default --gpu --start_frame 10 --end_frame 50 --debug
```
This command will run segmentation on the frames 10 to 50 of *data/movie.nd2* using the 'cyto' model, and save the results as a TIFF stack in *results/segmented.tiff*. It will also save additional files with flows and probabilities for debugging purposes.

The model output consists of a generated TIF file saved at the user specified location. The output file contains masks for each identified cell separated by integer level. For example, background is indicated by a 0 value, the first cell identified is identified with a 1 value in all pixels containing that cell, and so on for each identified cell. The base Cellpose framework structures output in this way to handle the condition where two segmented cells share a pixel on an edge, as if a single value was used for segmented cells, downstream analysis would treat those cells as merged into a single cell.

## Building a distributable binary

To build a distributable package for your operating system, you'll need to install the dependencies listed above, as well as [pyinstaller](https://www.pyinstaller.org/). Note that you'll need to be on a windows system for this to work. Then, run the following command in the root directory of this repository:
```bash
pyinstaller --onefile run_segmentation_backup.py
```
This will create a single executable file in the *dist* folder. You can then run this executable from the command line with the same arguments as above. If you are on Windows, this will be an exe file, and if you are on Linux, this will be an ELF file. On Mac, this will be a DMG file that should work.
