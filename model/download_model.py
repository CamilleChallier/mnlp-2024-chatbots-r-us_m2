import argparse
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from models.model_dpo import AutoDPOModelForSeq2SeqLM, AutoDPOModelForCausalLM
from datasets import load_dataset
import os


def main(args):
    # Download the pre-trained model and tokenizer from the Hub
    if args.seq2seq_or_causal == 'seq2seq':
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        model_wrapper = AutoDPOModelForSeq2SeqLM(pretrained_model=model, beta=0.1, tokenizer_path = args.model_name, train_dataset_path = "./datasets/dpo_M1_15052024_random_pair_train.jsonl")
    elif args.seq2seq_or_causal == 'causal':
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        model_wrapper = AutoDPOModelForCausalLM(pretrained_model=model, beta=0.1, tokenizer_path = args.model_name, train_dataset_path = "./datasets/dpo_M1_15052024_random_pair_train.jsonl")
    else : 
        raise(ValueError, f"`seq2seq_or_causal` variable should be eithr `seq2seq` or `causal`. Got {args.seq2seq_or_causal} instead.")
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    checkpoint_dir = os.path.join("./checkpoints", args.model_name.split('/')[-1])
    model_wrapper.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to load a model.")
    parser.add_argument('--model_name', type=str, default="google/flan-t5-small", help="Name of the pre-trained model from the Hugging Face Hub")
    parser.add_argument('--seq2seq_or_causal', type=str, default='seq2seq')

    args = parser.parse_args()
    main(args)