from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, GPT2Model, GPT2Tokenizer
from models.model_dpo import AutoDPOModelForSeq2SeqLM, AutoDPOModelForCausalLM
from datasets import load_dataset
import os
import argparse

from peft import PeftModel, PeftConfig, LoraConfig, TaskType, get_peft_model


def main(args):
    checkpoint_dir = os.path.join("./checkpoints", args.model_name)
    
    peft_config = LoraConfig(
        r=args.lora_r, # Rank
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
    )
    
    if args.seq2seq_or_causal == 'seq2seq':
        ref_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
        model = get_peft_model(model, peft_config)
        model_wrapper = AutoDPOModelForSeq2SeqLM.from_pretrained(model, beta=args.beta, train_dataset_path = args.train_dataset, save_name = args.save_name, lr=args.lr, eval_dataset=args.eval_dataset, batch_size =args.batch_size, max_length =args.max_length, ref_model=ref_model)
    # elif args.seq2seq_or_causal == 'causal':
    #     model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    #     model = get_peft_model(model, peft_config)
    #     model_wrapper = AutoDPOModelForCausalLM.from_pretrained(checkpoint_dir, beta=args.beta, train_dataset_path = args.train_dataset, save_name = args.save_name, lr=args.lr, eval_dataset=args.eval_dataset)
    else:
        raise(ValueError, f"`seq2seq_or_causal` variable should be `seq2seq`. Got {args.seq2seq_or_causal} instead.")
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    
    model_wrapper.train()
    
    if args.save_name is None:
        args.save_name = args.model_name + '_dpo'
    
    save_dir = os.path.join('./checkpoints', args.save_name)
    model_wrapper.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to train and save a DPO model.")
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--train_dataset', nargs='+')
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--seq2seq_or_causal', type=str, default='seq2seq')
    parser.add_argument('--eval_dataset', nargs='+')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_target_modules', nargs='+', default=['q', 'v'])

    args = parser.parse_args()
    main(args)