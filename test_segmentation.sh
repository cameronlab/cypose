#!/bin/bash

# You'll need to add your own path to the nd2 file
# python run_segmentation.py \
    #        --input_file channelred,gfp,cy5,rfp,bfp_seq0000_0005.nd2 \
    #        --output_file test.tif --gpu --debug \
    #        --model ./models/7002_CAH_Default
# python run_segmentation.py \
		# 			 --input_file "/Users/zachmaas/Desktop/2020.3.5_ana33047_minusn_0003_Crop.nd2" \
		# 			 --output_file "/Users/zachmaas/Desktop/masks_ours.tif" --gpu \
		# 			 --model ./models/merged_data_longrun --start_frame 200 --end_frame 205 \
		# 			 --denoise --flow_threshold 0.2
# python run_segmentation.py \
		# 			 --input_file "/Users/zachmaas/Desktop/2020.3.5_ana33047_minusn_0003_Crop.nd2" \
		# 			 --output_file "/Users/zachmaas/Desktop/masks_cp2.tif" --gpu \
		# 			 --model cyto2 --start_frame 200 --end_frame 205
# python run_segmentation.py \
		# 			 --input_file "/Users/zachmaas/Desktop/2020.3.5_ana33047_minusn_0003_Crop.nd2" \
		# 			 --output_file "/Users/zachmaas/Desktop/masks_cp3.tif" --gpu \
		# 			 --model cyto3 --start_frame 200 --end_frame 205

# parallel -j 1 python run_segmentation.py \
		# 				 --input_file "/Users/zachmaas/Desktop/2020.3.5_ana33047_minusn_0003_Crop.nd2" \
		# 				 --output_file "/Users/zachmaas/Desktop/masks_{}.tif" --gpu \
		# 				 --model ./models/merged_data_longrun --start_frame 200 --end_frame 201 \
		# 				 --flow_threshold {} ::: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
# parallel -j 1 python run_segmentation.py \
		# 				 --input_file "/Users/zachmaas/Desktop/2020.3.5_ana33047_minusn_0003_Crop.nd2" \
		# 				 --output_file "/Users/zachmaas/Desktop/masks_{}.tif" --gpu \
		# 				 --model ./models/merged_data_longrun --start_frame 200 --end_frame 201 \
		# 				 --niter {} ::: 100 500 1000 5000 10000

python run_segmentation.py \
			 --input_file "/Users/zachmaas/Desktop/AnaFiles/20210709_Ana_-N_to_-N_channelbf,cy5,cfp,rfp_seq0000_0001.nd2" \
			 --output_file "/Users/zachmaas/Desktop/masks_bench_5.tif" --gpu \
			 --model ./models/corrected_data_300epoch_2ch --start_frame 90 --end_frame 94 \
			 --flow_threshold 0.75 --niter 1000 --size 5
python run_segmentation.py \
			 --input_file "/Users/zachmaas/Desktop/AnaFiles/20210709_Ana_-N_to_-N_channelbf,cy5,cfp,rfp_seq0000_0001.nd2" \
			 --output_file "/Users/zachmaas/Desktop/masks_bench_10.tif" --gpu \
			 --model ./models/corrected_data_300epoch_2ch --start_frame 90 --end_frame 94 \
			 --flow_threshold 0.75 --niter 1000 --size 10
python run_segmentation.py \
			 --input_file "/Users/zachmaas/Desktop/AnaFiles/20210709_Ana_-N_to_-N_channelbf,cy5,cfp,rfp_seq0000_0001.nd2" \
			 --output_file "/Users/zachmaas/Desktop/masks_bench_15.tif" --gpu \
			 --model ./models/corrected_data_300epoch_2ch --start_frame 90 --end_frame 94 \
			 --flow_threshold 0.75 --niter 1000 --size 15
python run_segmentation.py \
			 --input_file "/Users/zachmaas/Desktop/AnaFiles/20210709_Ana_-N_to_-N_channelbf,cy5,cfp,rfp_seq0000_0001.nd2" \
			 --output_file "/Users/zachmaas/Desktop/masks_bench_20.tif" --gpu \
			 --model ./models/corrected_data_300epoch_2ch --start_frame 90 --end_frame 94 \
			 --flow_threshold 0.75 --niter 1000 --size 20
