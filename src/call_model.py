from transformers import TextStreamer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

relative_path = "src/finetune/checkpoint"
absolute_path = os.path.abspath(relative_path)

BASE_MODEL_NAME = "arcee-ai/Arcee-VyLinh"
ADAPTER_PATH = absolute_path

def load_model():
    # Base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    
    #Adapter LoRa
    finetuned_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    return finetuned_model, tokenizer


finetuned_model, tokenizer = load_model()


system_prompt = "Bạn là trợ lý y tế chuyên nghiệp, trả lời câu hỏi một cách chi tiết và chính xác."
question = str(input("Nhập câu hỏi: "))
prompt = f"[INST] <<SYS>>\\n{system_prompt}\\n<</SYS>>\\n\\n{question} [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,        
    skip_special_tokens=True 
)
outputs = finetuned_model.generate(
    **inputs,
    max_new_tokens=512,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    streamer=streamer
)