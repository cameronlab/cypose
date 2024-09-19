#!/bin/bash

# You'll need to add your own path to the nd2 file
python run_segmentation.py \
			 --input_file channelred,gfp,cy5,rfp,bfp_seq0000_0005.nd2 \
			 --output_file test.tif --gpu --debug \
			 --model ./models/7002_CAH_Default
