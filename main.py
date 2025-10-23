# main.py
import sys
# Bỏ 'client' vì không cần cho rewriter nữa
from src.generative_model import call_model, conversation_memory
from src.config import initialize_once, search_index

# XÓA: import from src.query_rewriter

# Khởi tạo 1 lần duy nhất
initialize_once()

print("Hệ thống Chatbot (Ollama API) đã sẵn sàng!")
session_id = "user_session_main"

while True:
    try:
        question = str(input("\nUser (gõ 'q' để thoát): "))
        if question.lower() == 'q':
            break

        if not question.strip():
            print("Hãy hỏi tôi gì đó!")
            continue

        # === SỬA LOGIC QUERY ===
        # Lấy 4 tin nhắn cuối làm ngữ cảnh
        chat_history = conversation_memory.chat_memory.messages[-4:]

        # Tạo query có ngữ cảnh (Contextual Query) cho RAG
        contextual_query = question
        if chat_history:
            # Ghép lịch sử và câu hỏi mới lại
            history_str = "\n".join([
                f"{'User' if msg.type == 'human' else 'Bot'}: {msg.content[:100]}..."
                for msg in chat_history
            ])
            contextual_query = f"Lịch sử:\n{history_str}\n\nCâu hỏi mới: {question}"
            print(f"[Debug] Đang dùng query có ngữ cảnh để tìm kiếm...")
        else:
            print(f"[Debug] Lịch sử rỗng, dùng câu hỏi gốc để tìm kiếm.")

        # 1. Dùng query có ngữ cảnh để tìm kiếm vector
        docs, similarity_score = search_index(contextual_query)

        # 2. Gọi model với câu hỏi GỐC (để model tự xử lý ngữ cảnh)
        #    (Hàm call_model bây giờ chỉ nhận 1 'question')
        call_model(
            question=question,  # Chỉ dùng câu hỏi gốc
            docs=docs,
            session_id=session_id,
            similarity_score=similarity_score
        )
        # ======================

    except KeyboardInterrupt:
        print("\nĐang thoát...")
        sys.exit()
    except Exception as e:
        print(f"\n[LỖI]: {e}")
        import traceback

        traceback.print_exc()