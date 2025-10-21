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

data_name = 'Dqdung205/medical-vietnamese-qa'
model_name = "AITeamVN/Vi-Qwen2-1.5B-RAG"
max_length = 512

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
    # device_map="auto"
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
    # Sử dụng cấu hình tiêu chuẩn
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

print("Đang áp dụng LoRA...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("Dqdung205/medical-vietnamese-qa", split="train")

def format_and_tokenize(examples):
    chats = []
    for question, answer in zip(examples['question'], examples['answer']):
        messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        chats.append(messages)
    
    return {"input_ids": tokenizer.apply_chat_template(chats, truncation=True, max_length=max_length)}

# Áp dụng hàm và xóa cột gốc 'question', 'answer'
tokenized_dataset = dataset.map(
    format_and_tokenize,
    batched=True,
    remove_columns=['question', 'answer']
)
training_data = tokenized_dataset.train_test_split(test_size=0.1)

training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
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
try:
    model_device = next(trainer.model.parameters()).device
    print(f"Model đã sẵn sàng và đang nằm trên thiết bị: {model_device}")
except Exception as e:
    print(f"Không thể xác định thiết bị của model. Lỗi: {e}")
# Bắt đầu training
print("Bắt đầu training...")
trainer.train(resume_from_checkpoint=True)
