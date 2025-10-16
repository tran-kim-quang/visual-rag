from src.generative_model import call_model, load_model
from src.config import search_index
print("Đang khởi động model...")
finetune, tokenize = load_model()
print("Đã load xong!")
while True:
    query = str(input("User: "))
    if query=="q":
        break
    if not query:
        print("Hãy hỏi tôi gì đó!")
    context = search_index(query)
    call_model(query, context, finetune, tokenize)