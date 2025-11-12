# main.py
import sys
from dotenv import load_dotenv
import os 

from src.generative_model import call_model, conversation_memory
from src.config import initialize_once, search_index
from src.query_rewriter import rewrite_query
from src.data_processor import MedicalDataProcessor

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 

if not OPENROUTER_API_KEY:
    print("\n[LỖI CẤU HÌNH]: Biến OPENROUTER_API_KEY chưa được đặt trong file .env!")
    sys.exit(1)

initialize_once(api_key=OPENROUTER_API_KEY)
processor = MedicalDataProcessor()

print("Hệ thống Chatbot (OpenRouter API) đã sẵn sàng!")

session_id = "user_session_main"
MAX_HISTORY = 10

while True:
    try:
        question = str(input("\nUser (gõ 'q' để thoát): "))
        if question.lower() == 'q':
            break

        if not question.strip():
            print("Hãy hỏi tôi gì đó!")
            continue
        
        # Preprocess
        question = processor.preprocess_query(question)
        if not question:
            print("Câu hỏi không hợp lệ!")
            continue

        chat_history = conversation_memory.chat_memory.messages
        
        # Trim history
        if len(chat_history) > MAX_HISTORY:
            conversation_memory.chat_memory.messages = chat_history[-MAX_HISTORY:]
            chat_history = conversation_memory.chat_memory.messages
        
        recent_history = chat_history[-4:]

        # REWRITE QUERY
        if recent_history:
            rewritten_query = rewrite_query(question, recent_history)
            print(f"[Query Rewrite] Original: {question}")
            print(f"[Query Rewrite] Rewritten: {rewritten_query}")
            if not rewritten_query or len(rewritten_query) < 3:
                rewritten_query = question
        else:
            rewritten_query = question
            print(f"[Debug] Lịch sử rỗng, dùng câu hỏi gốc")

        # Search
        docs, similarity_score = search_index(rewritten_query, k=15)
        
        if not docs:
            print(f"[Fallback] Không tìm được docs, thử query gốc...")
            docs, similarity_score = search_index(question, k=15)
        elif similarity_score < 0.3:
            print(f"[Fallback] Rerank score thấp ({similarity_score:.2f}), thử query gốc...")
            docs_fallback, sim_fallback = search_index(question, k=15)
            if sim_fallback > similarity_score:
                docs = docs_fallback
                similarity_score = sim_fallback

        call_model(
            question=question,
            docs=docs,
            session_id=session_id,
            similarity_score=similarity_score
        )

    except KeyboardInterrupt:
        print("\nĐang thoát...")
        sys.exit()
    except Exception as e:
        print(f"\n[LỖI]: {e}")
        import traceback
        traceback.print_exc()