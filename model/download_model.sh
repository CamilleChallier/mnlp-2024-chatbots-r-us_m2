#!/bin/bash -l
#SBATCH --chdir /scratch/izar/challier/project-m2-2024-chatbots-r-us/model
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 16G
#SBATCH --time 0:07:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552

echo STARTING AT `date`
echo IN `pwd`
nvidia-smi

python3 -u download_model.py --model_name "google/flan-t5-small" --seq2seq_or_causal "seq2seq" # 77M params
python3 -u download_model.py --model_name "google/flan-t5-base" --seq2seq_or_causal "seq2seq" # 248M params
python3 -u download_model.py --model_name "google/flan-t5-large" --seq2seq_or_causal "seq2seq" # 783M params
python3 -u download_model.py --model_name "facebook/bart-base" --seq2seq_or_causal "seq2seq" # ~140M
python3 -u download_model.py --model_name "facebook/bart-large" --seq2seq_or_causal "seq2seq" # ~400M
#python3 -u download_model.py --model_name "openai-community/gpt2" --seq2seq_or_causal "causal" # 137M params
#python3 -u download_model.py --model_name "openai-community/gpt2-large" --seq2seq_or_causal "causal" # 812M params

echo FINISHED at `date`