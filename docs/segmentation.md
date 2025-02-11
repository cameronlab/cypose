---
title: Segmentation
---

The segementation algorithms provided by CyPose are derivative of those originally implemented in [CellPose](https://cellpose.readthedocs.io/en/latest/ "CellPose Documentation"). Over the course of trying to leverage CellPose (and its derivative OmniPose), we found that the provided models as well as the code itself were not well-suited to our use case. Consequently, we have fine-tuned specialized models for Cyanobacterial cells and provide extensive documentation on pitfalls that one might encounter when using these models or the CellPose base models and library.

!!! warning
	The implementation of CellPose provided in their GUI will consistently produce different results than the implementation provided in the underlying Python library, and the differences between the two are not well documented. If your segmentation results are worse than expected, check to see if the GUI implementation produces better results. The GUI and underlying library are likely to diverge as CellPose development continues.

Segmentation is performed using the `run_segmentation` module, found in `run_segmetation.py`. Input can be in the form of either an ND2 image/movie or a TIF image/movie, and output will be a TIF image/movie containing the segmented masks. The `run_segmentation` module is designed to be run from the command line, and can be run as follows:
```bash
python run_segmentation.py \
			 --input_file "movie_name" \
			 --output_file "mask_name" \
			 --gpu \
			 --model ./models/model_name \
			 --start_frame 200 \
			 --end_frame 205
```

The following arguments are available:

- **`--input_file "movie_name"`**: Specifies the input movie file. ND2 or TIF formatted.
- **`--output_file "mask_name"`**: Specifies the output mask file. TIF formatted.
- **`--gpu`**: Indicates that the script should use the GPU. Device is automatically inferred (MPS for M1 OSX and CUDA for Nvidia GPUs).
- **`--model ./models/model_name`**: Specifies the path to the model file. Can also be any built-in Cellpose 2 or 3 model name.
- **`--start_frame 200`**: Specifies the start frame for the segmentation.
- **`--end_frame 205`**: Specifies the end frame for the segmentation.

Additionally, three experimental options are provided:

- **--debug**: Specifies a debugging run, which will also save probabilities and flows for the model.
- **--denoise**: Preliminary _experimental_ support for Cellpose 3's built-in denoising model. We do not currently reccommend using this - if you are going to be doing downstream analysis on your data, don't use low quality data.
- **--niter**: Run the model for n-iterations for each frame to generate higher quality frames. This method is derived from Omnipose and can sometimes return better results with high iterations (2000) on bacterial imagery.
- **-flow_threshold**: Run the model with a different flow threshold for classifying cells. We use a default of 0.75, compared to the Cellpose default of 0.4, and find generally better results with this approach.

The model output consists of a generated TIF file saved at the user specified location. The output file contains masks for each identified cell separated by integer level. For example, background is indicated by a 0 value, the first cell identified is identified with a 1 value in all pixels containing that cell, and so on for each identified cell. The base Cellpose framework structures output in this way to handle the condition where two segmented cells share a pixel on an edge, as if a single value was used for segmented cells, downstream analysis would treat those cells as merged into a single cell.

If generated, the flow-threshold and probability output files contain data in a generic float representation or bounded float (between 0 and 1), respectively. These files are useful for manual verification of model behavior.
