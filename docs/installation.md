# Installing and Running CyPose

To install CyPose, first clone the repository from GitHub:
```bash
git clone https://github.com/cameronlab/cypose.git
```
You will need to set up a virtual environment, using either virtualenv or conda (or another tool of your choosing). Then, install poetry and the project dependencies:
```bash
python3 -m pip poetry
poetry install
```

This project is structured so that segmentation is easy if you have an input movie in TIF format. See the `test_segmentation.sh` file for an example of how to run segmentation using a provided model:
```bash
python run_segmentation.py \
		 		   --input_file "/path/to/tif" \
		 		   --output_file "/path/to/out_masks.tif" \
				   --model ./models/your_model --start_frame 0 --end_frame 224
```
By default, this script will segment the input file and output a TIF formatted mask file suitable for use as input for downstream tools like [CyAn](https://github.com/Biofrontiers-ALMC/CyAn).

Models are provided in one of two formats:

- Single channel movies, which are designed to be used against a brightfield image. These are the most versatile.
- Two channel movies, which are designed for movies with brightfield and chlorphyll images. In the case of some strains, we see better performance using the chlorphyll channel by allowing the model to more reliably learn the morphological differences between distinct celltypes.

By default, our best models are listed by the strain name. You can also choose to use a pre-built cellpose model (e.g. 'cyto2' or 'cyto3') if applying these tools to a celltype that a well performing fine-tuned model is not available for.
