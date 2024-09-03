#!/bin/bash -l
#SBATCH --chdir /scratch/izar/monteith/project-m2-2024-chatbots-r-us/model
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --mem 16G
#SBATCH --time 0:30:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552

echo STARTING AT `date`
echo IN `pwd`
nvidia-smi

python3 -u evaluator.py

# python3 -u evaluator_custom.py --main_config_path ./main_config/main_config0.yaml --metrics_save_name metrics/flan-t5-small_orca_metrics.json
# python3 -u evaluator_custom.py --main_config_path ./main_config/main_config1.yaml --metrics_save_name metrics/flan-t5-small_webgpt_metrics.json
# python3 -u evaluator_custom.py --main_config_path ./main_config/main_config2.yaml --metrics_save_name metrics/metrics_flan-t5-small_M1.json
# python3 -u evaluator_custom.py --main_config_path ./main_config/main_config3.yaml --metrics_save_name metrics/flan-t5-base-finetuned_orca_metrics.json
# python3 -u evaluator_custom.py --main_config_path ./main_config/main_config4.yaml --metrics_save_name metrics/flan-t5-base-finetuned_webgpt_metrics.json
# python3 -u evaluator_custom.py --main_config_path ./main_config/main_config5.yaml --metrics_save_name metrics/flan-t5-base-finetuned_M1_metrics.json
# python3 -u evaluator_custom.py --main_config_path ./main_config/main_config6.yaml --metrics_save_name metrics/metrics_flan-t5-large-finetuned_orca2.json
# python3 -u evaluator_custom.py --main_config_path ./main_config/main_config7.yaml --metrics_save_name metrics/metrics_flan-t5-large-finetuned_webgpt2.json
# python3 -u evaluator_custom.py --main_config_path ./main_config/main_config8.yaml --metrics_save_name metrics/metrics_flan-t5-large-finetuned_M12.json

# python3 -u evaluator_custom.py --main_config_path ./main_config/main_config9.yaml --metrics_save_name metrics/metrics_bart-base_orca.json
# python3 -u evaluator_custom.py --main_config_path ./main_config/main_config10.yaml --metrics_save_name metrics/metrics_bart-base_webgpt.json
# python3 -u evaluator_custom.py --main_config_path ./main_config/main_config11.yaml --metrics_save_name metrics/metrics_bart-base_M1.json
# python3 -u evaluator_custom.py --main_config_path ./main_config/main_config12.yaml --metrics_save_name metrics/metrics_bart-large_orca.json
# python3 -u evaluator_custom.py --main_config_path ./main_config/main_config13.yaml --metrics_save_name metrics/metrics_bart-large_webgpt.json
# python3 -u evaluator_custom.py --main_config_path ./main_config/main_config14.yaml --metrics_save_name metrics/metrics_bart-large_M1.json

# python3 -u evaluator_custom.py --main_config_path ./main_config/main_config00.yaml --metrics_save_name metrics/metrics_flan-t5-base_all2.json

echo FINISHED at `date`