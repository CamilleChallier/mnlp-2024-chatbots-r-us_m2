import os
import yaml
import glob
import torch
import logging
import evaluate
import numpy as np

from torch.utils.data import DataLoader

from typing import Dict, Union, List, Any, Tuple
from transformers import PreTrainedTokenizerBase, AutoTokenizer, AutoModelForSeq2SeqLM

from utils import read_jsonl, write_json
from models.model_base import PreTrainedModelWrapper
from models.model_dpo import AutoDPOModelForCausalLM, AutoDPOModelForSeq2SeqLM
from peft import PeftModel, PeftConfig, LoraConfig, TaskType, get_peft_model
import argparse


def main(args):
    # Load the main configuration file
    main_config = {}
    with open(args.main_config_path) as f:
        main_config = yaml.safe_load(f)

    # Load the task type to identify the model class
    task_type = main_config.get("task_type", "causal_lm")

    # Load the evaluation methods and the required paths
    eval_method = main_config.get("eval_method", ["mcqa"])
    policy_model_path = main_config["policy_model_path"]
    reference_model_path = main_config["reference_model_path"]
    test_data_path = main_config["test_data_path"]
    
    # Load peft config for pre-trained checkpoint etc.
    config = PeftConfig.from_pretrained(policy_model_path)
    
    # load base LLM model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    # tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    ref_model = AutoModelForSeq2SeqLM.from_pretrained(reference_model_path)
    
    # Load the Lora model
    model = PeftModel.from_pretrained(model, policy_model_path)
    
    model_wrapper = AutoDPOModelForSeq2SeqLM.from_pretrained(model, ref_model =ref_model, eval_dataset=test_data_path)
    
    accuracy = model_wrapper.evaluate()

    metrics = {
        "team_name": "Chatbots-R-Us",
        "task_type": "seq2seq",
        "policy_reward_accuracy": accuracy
    }
    write_json(metrics, args.metrics_save_name)
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a DPO model")
    parser.add_argument('--main_config_path', type=str, default='main_config.yaml')
    parser.add_argument('--metrics_save_name', type=str, default='metrics.json')
    args = parser.parse_args()
    main(args)