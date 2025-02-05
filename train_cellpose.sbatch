#!/bin/bash
#SBATCH --output=/scratch/Users/zama8258/e_and_o/%x_%j.out
#SBATCH --error=/scratch/Users/zama8258/e_and_o/%x_%j.err
#SBATCH -p nvidia-a100
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --mail-user=zama8258@colorado.edu

set -euxo pipefail

trainDir=/scratch/Users/zama8258/ZachML/corrected_data
model="cyto2"

cd "$trainDir" || exit 1
python -m cellpose --train --use_gpu \
        --verbose --chan 0 --chan2 1 \
        --dir "$trainDir" \
        --mask_filter _labeled \
        --img_filter _img \
        --pretrained_model "$model" \
        --learning_rate 0.00001 --weight_decay 0.0001 \
        --n_epochs 500 --save_every 50
