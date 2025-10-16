from transformers import TextStreamer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv
import os
load_dotenv()

relative_path = os.getenv('CHECKPOINT')
absolute_path = os.path.abspath(relative_path)

BASE_MODEL_NAME = os.getenv('BASE_MODEL_NAME')
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

def call_model(question, context, finetuned_model, tokenizer):
    # finetuned_model, tokenizer = load_model()

    system_prompt = "Bạn là trợ lý y tế chuyên nghiệp. Hãy sử dụng ngữ cảnh được cung cấp để trả lời câu hỏi của người dùng một cách chi tiết và chính xác."

    prompt = f"""[INST] <<SYS>>
        {system_prompt}
        <</SYS>>
        Dựa vào ngữ cảnh sau đây để trả lời câu hỏi ở cuối.
        Nếu ngữ cảnh không chứa thông tin, hãy trả lời là "Tôi không tìm thấy thông tin về vấn đề này trong tài liệu."
        --- NGỮ CẢNH ---
        {context}
        --- HẾT NGỮ CẢNH ---
        CÂU HỎI: {question} [/INST]"""
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
