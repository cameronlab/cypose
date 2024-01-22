#!/bin/bash
which python
# You'll need to add your own path to the nd2 file
python run_segmentation.py \
--input_file 'E:\ZachML\Chris Data\20210709_Ana_-N_to_-N\channelbf,cy5,cfp,rfp_seq0000_0000.nd2' \
--output_file 'E:\ZachML\Chris Data\20210709_Ana_-N_to_-N\20210709_Ana_-N_to_-N_channelbf,cy5,cfp,rfp_seq0000_0000_Mask.tif' \
--start_frame 0 \
--end_frame 53 \
--gpu \
--model ./models/chris_prelim_iter1_2chan_v2
