#!/bin/bash
#SBATCH --job-name=o5_3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=syed-mahbubul.huq.2@city.ac.uk
#SBATCH --partition=preemptgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100_80g:1
#SBATCH -o outputs/%x_%j.o
#SBATCH -e outputs/%x_%j.e

# Ensure output directory exists so SLURM can write logs
mkdir -p outputs

# Environment Setup
source /users/adhp263/sharedscratch/miniconda3/etc/profile.d/conda.sh
conda activate diplomacy_env

# --- CRITICAL FIXES FOR MEMORY ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Run extraction for each template

MODEL="/users/adhp263/sharedscratch/models/o5"


python base.py --instances all --split test --model "$MODEL"
