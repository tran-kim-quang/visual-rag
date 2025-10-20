import json
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

def format_medical_qa():
    print("Đang tải dataset 'hungnm/vietnamese-medical-qa'...")
    dataset = load_dataset("hungnm/vietnamese-medical-qa", split="train")
    output_file = "finetune/formatted_medical_qa.jsonl"
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
dataset = load_dataset("json", data_files="finetune/formatted_medical_qa.jsonl", split="train")

max_length = 1024

def tokenize_function(examples):
    # Combine question and answer into a single text field for each message
    # Assuming each 'messages' entry is a list of dicts like [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
    text = []
    for conversation in examples['messages']:
        # Join the content of each message in the conversation
        conversation_text = " ".join([msg['content'] for msg in conversation])
        text.append(conversation_text)

    tokenized_output = tokenizer(text, padding="max_length", truncation=True, max_length=max_length)
    return {
        "input_ids": tokenized_output["input_ids"],
        "attention_mask": tokenized_output["attention_mask"],
    }

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['messages'])


training_data = tokenized_datasets.train_test_split(test_size=0.2)
# Tên model trên Hugging Face Hub
model_name = "bigscience/bloomz-560m"

# --- 2. Tải Model và Tokenizer với Unsloth (đơn giản hơn nhiều) ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_length,
    dtype = None, 
    load_in_4bit = True, 
)

# --- Thêm các adapter LoRA vào model ---
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank của LoRA. 8 hoặc 16 là đủ dùng
    lora_alpha = 32,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 3407,
    target_modules = ["query_key_value"], # Các module bạn muốn áp dụng LoRA
)

# --- Tải dữ liệu (giữ nguyên như cũ) ---
dataset = load_dataset("json", data_files="formatted_medical_qa.jsonl", split="train")
training_data = dataset.train_test_split(test_size=0.1)

# --- Thiết lập TrainingArguments (gần như giữ nguyên) ---
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
    optim="adamw_8bit", # Optimizer tối ưu của Unsloth
    
    # Các tham số đánh giá
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    evaluation_strategy="steps",
    eval_steps=100,
)

# --- Khởi tạo Trainer (giữ nguyên) ---
trainer = SFTTrainer(
    model=model,
    train_dataset=training_data["train"],
    eval_dataset=training_data["test"],
    dataset_text_field="text", # Giả sử cột dữ liệu của bạn tên là "text"
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_arguments,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Bắt đầu training
trainer.train()
