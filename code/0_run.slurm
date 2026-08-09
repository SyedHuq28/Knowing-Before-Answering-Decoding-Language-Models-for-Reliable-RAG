#!/bin/bash
#SBATCH --job-name=o5
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
# Optional: force-compliance diagnostic

python 2.py --model "$MODEL" --template P0 --split train
python 2.py --model "$MODEL" --template P0 --split val
python 2.py --model "$MODEL" --template P0 --split test



python 3.py

python 5a.py --model "$MODEL" --split test
python 6.py --train_mode train
python 6.py --train_mode trainval
python 7_judge2.py --glob "results/ablation_val_a*.json" --out_dir "results/rescored_val"
python 7_judge2.py --glob "results/ablation_test_a*.json" --out_dir "results/rescored_test"


python 11.py
