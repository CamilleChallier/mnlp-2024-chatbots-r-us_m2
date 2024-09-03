#!/bin/bash -l
#SBATCH --chdir /scratch/izar/monteith/project-m2-2024-chatbots-r-us/model/
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 16G
#SBATCH --time 3:00:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552

echo STARTING AT `date`
echo IN `pwd`
nvidia-smi

python3 -u dpo_tuning.py --model_name "flan-t5-small" --train_dataset ./datasets/dpo/train/dpo_M1_all_pairs_length-sup-200-char_train.jsonl ./datasets/dpo/train/dpo_orca_train.jsonl ./datasets/dpo/train/dpo_webgpt_comparaisons_train.jsonl --save_name "flan-t5-small_LoRA_dpo-M1-orca-webgpt_bs-8_max-len-512_lr-1e-4" --beta 0.1 --seq2seq_or_causal "seq2seq" --eval_dataset ./datasets/dpo/test/dpo_M1_all_pairs_length-sup-200-char_test.jsonl ./datasets/dpo/test/dpo_orca_test.jsonl ./datasets/dpo/test/dpo_webgpt_comparaisons_test.jsonl --batch_size 8 --max_length 512 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --lr 0.0001 --lora_target_modules q v

echo FINISHED at `date`