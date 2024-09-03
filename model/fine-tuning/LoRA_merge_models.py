from transformers import AutoModelForCausalLM, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel


# ===========================================
# Merge the base model with the PEFT model
# ===========================================

#base_model_path = "/home/ckalberm/project-m2-2024-chatbots-r-us/models/flan-t5-base"
base_model_path = "google/flan-t5-base"
peft_model_path = "models/flan-t5-base_LoRA_dpo-M1-orca-webgpt_bs-4_max-len-512_lr-1e-4\checkpoint-34191"
#peft_model_path = "/home/ckalberm/project-m2-2024-chatbots-r-us/models/flan-t5-base_LoRA_dpo-M1-orca-webgpt_bs-4_max-len-512_lr-1e-4"
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(base_model, peft_model_path)
tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
merged_model = model.merge_and_unload()

# ===========================================
# Save the merged model for later use
# ===========================================

# save the merged model
merged_model_path = "models/flan-t5-base_LoRA_merged"
merged_model.save_pretrained(merged_model_path)
# # save the tokenizer
tokenizer.save_pretrained(merged_model_path)


