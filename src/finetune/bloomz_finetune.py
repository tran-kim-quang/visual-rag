import json
import torch
from datasets import load_dataset
from transformers import (
    TrainingArguments, 
    EarlyStoppingCallback, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

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
    print(f"Đã xử lý {count} bản ghi. File: {output_file}")

# format_medical_qa()
dataset = load_dataset("json", data_files="formatted_medical_qa.jsonl", split="train")

max_length = 1024

# Tên model trên Hugging Face Hub
model_name = "bigscience/bloomz-560m"

# Tải model với BitsAndBytes 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)

print(f"Đang tải model {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)

print("Đang tải tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Cấu hình LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["query_key_value"],
)

print("Đang áp dụng LoRA...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Định nghĩa hàm preprocessing
def preprocess_function(examples):
    # Kết hợp message thành text
    texts = []
    for messages in examples['messages']:
        text = ""
        for message in messages:
            role = message['role']
            content = message['content']
            if role == "user":
                text += f"Câu hỏi: {content}\n"
            elif role == "assistant":
                text += f"Trả lời: {content}\n"
        texts.append(text)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        return_tensors=None
    )
    return tokenized

# Chuẩn bị dữ liệu
print("Đang xử lý dữ liệu...")
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=['messages'],
    desc="Tokenizing"
)

# Chia train/test
training_data = tokenized_dataset.train_test_split(test_size=0.1)

# Cấu hình training
training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    save_steps=100,
    optim="paged_adamw_32bit",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_total_limit=2,
    report_to="none",
)

# Khởi tạo trainer
print("Đang khởi tạo trainer...")
trainer = Trainer(
    model=model,
    train_dataset=training_data["train"],
    eval_dataset=training_data["test"],
    tokenizer=tokenizer,
    args=training_arguments,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Bắt đầu training
print("Bắt đầu training...")
trainer.train()
