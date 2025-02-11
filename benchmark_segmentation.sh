#!/bin/bash
#SBATCH --output=/scratch/Users/zama8258/e_and_o/%x_%j.out
#SBATCH --error=/scratch/Users/zama8258/e_and_o/%x_%j.err
#SBATCH -p nvidia-a100
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=16gb
#SBATCH --mail-user=zama8258@colorado.edu

set -euxo pipefail

# function finish {
# 		if command -v apprise &> /dev/null
# 			 then
# 					 apprise -t "Fiji Run Complete" -b "Benchmark Segmentation"
# 					 exit
# 		fi
# }
# trap finish EXIT

# pushd /scratch/Users/zama8258/cyanosegment || exit

# root_dir="/Users/zachmaas/Desktop/benchmark/20240428_Benchmark"
# root_dir="/scratch/Users/zama8258/ZachML/benchmark"
# data="$root_dir/channelred,gfp,cy5,rfp,bfp_seq0000_0003.nd2"
root_dir="/Users/zachmaas/Desktop/AnaFiles/"
data="$root_dir/20210709_Ana_-N_to_-N_channelbf,cy5,cfp,rfp_seq0000_0001.nd2"
# python run_segmentation_backup.py \
		# 			 --input_file "$data" \
		# 			 --output_file "$root_dir/masks_ours.tif" --gpu \
		# 			 --model ../models/7002_CAH_Default
python run_segmentation.py \
			 --input_file "$data" \
			 --output_file "$root_dir/masks_cyto2.tif" --gpu \
			 --model cyto2 --flow_threshold 0.4
python run_segmentation.py \
			 --input_file "$data" \
			 --output_file "$root_dir/masks_cyto3.tif" --gpu \
			 --model cyto3 --flow_threshold 0.4
python run_segmentation.py \
			 --input_file "$data" \
			 --output_file "$root_dir/masks_omni.tif" --gpu \
			 --model bact_fluor_cp3 --flow_threshold 0.4
