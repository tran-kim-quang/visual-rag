from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
from bloomz_finetune import get_peftModel
peft_model = get_peftModel()
training_args = TrainingArguments(
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    warmup_steps = 100,

    output_dir="./bloomz-medical-finetuned",
    num_train_epochs=3,
    learning_rate=3e-4,
    logging_steps=50,
    save_total_limit=2,

    remove_unused_columns=False, # Added this line

    optim = "adamw_8bit",
    eval_strategy = "steps",
    eval_steps = 100,
    save_strategy = "steps",
    save_steps = 100,

    load_best_model_at_end = True,
    metric_for_best_model = "eval_loss",
    greater_is_better = False,
)
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)
trainer.train()