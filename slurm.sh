#!/bin/sh
#SBATCH --job-name=worldocr_qwen25vl_test
#SBATCH --output=logs/worldocr_qwen25vl_test.out
#SBATCH --error=logs/worldocr_qwen25vl_test.err
#SBATCH --qos cscc-gpu-qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --mem=80GB


# create a fake script the keep sleep for the whole duration, 3 days job, with 4 gpus and 1 node, with 100GBs memory

# conda activate worldocr
cd /l/users/ahmed.heakl/worldocr
bash stall.sh


