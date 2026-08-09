#!/bin/bash
#SBATCH --job-name=o5_2
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

mkdir -p outputs results

source ~/.hpc_setup.sh

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false


python base_rag3.py \
  --model o5 \
  --model_path /users/adhp263/sharedscratch/models/o5 \
  --data instances_test.jsonl \


python rag_base4.py \
  --model o5 \
  --model_path /users/adhp263/sharedscratch/models/o5 \
  --data instances_test.jsonl \
  --output results/test.jsonl
