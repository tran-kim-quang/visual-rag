import json
from datasets import load_dataset

from peft import LoraConfig, get_peft_model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def format_medical_qa():
    print("Đang tải dataset 'hungnm/vietnamese-medical-qa'...")
    dataset = load_dataset("hungnm/vietnamese-medical-qa", split="train")
    output_file = "formatted_medical_qa.jsonl"
    print(f"Bắt đầu chuyển đổi và ghi ra file '{output_file}'...")

    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset:
            question = example["question"]
            answer = example["answer"]

            if not question or not answer:
                continue
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]

            json_object = {"messages": messages}

            f.write(json.dumps(json_object, ensure_ascii=False) + "\n")
            count += 1
    print(f"Dir: {output_file}")
# format_medical_qa()

model_name = "bigscience/bloomz-560m"

# Configure 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_compute_dtype=torch.float16, # You might need to adjust this based on your GPU
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load model with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# Cấu hình LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model, lora_config)

# num of parameters can training
peft_model.print_trainable_parameters()
def get_peftModel():
    return peft_model