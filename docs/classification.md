---
title: Classification
---

The segmentation model provided in CyPose has been hand-developed to work well with a variety of cyanobacterial cells, but requires more work to train a model on your data. You will need data containing your respective strains / cell types of interest as follows:

- An original movie in nd2 format to pull data from
- For each cell type, a tif format mask file containing masks specific to each cell type.

Files should be labeled as follows:

- A base name for all files from a given movie, used for the nd2. For example, `my_movie.nd2`
- Files with matching cell type labels added to the masks for that nd2. For example, `my_movie_cell1.tif`

So, for example, you might have the following:

- `my_movie.nd2`, `my_movie_cell1.tif`, `my_movie_cell2.tif`, `my_movie_cell3.tif`
- `my_movie2.nd2`, `my_movie2_cell1.tif`, `my_movie2_cell2.tif`, `my_movie2_cell3.tif`
- `my_movie3.nd2`, `my_movie3_cell1.tif`, `my_movie3_cell2.tif`, `my_movie3_cell3.tif`

!!! warning
	You must have the same set of channels in the same order in each movie used for training.

Once you have your data in this format, you can train a custom classifier model using the `train_classification_model.py` script, after making the following changes:

- Update the `root_dir` variable in the FastND2DataSet loader to point to the directory containing your data.
- In the ConvNetClassifier class, update the `num_channels` variable to match the number of channels in your movie and the `num_classes` variable to match the number of cell types you are training on.

Once that's done, you can run the script to train a model on your data. The script will save the model to a file, which you can then use to classify new data. To do this, you can use the run_classification.py script, which will classify all the cells in a given movie and save the results to a CSV file. This script can be run as follows:

```bash
python run_classification.py --model_path /path/to/model --data_dir /path/to/data
```

This will save the results to a file called `classification_results.csv` in the data directory. The results will contain the following columns:
```csv
frame,centroid_x,centroid_y,cell_type
```

This can then be used in any downstream analysis you wish to perform.
