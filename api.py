# api.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware  # Thêm dòng này

# Import các hàm cốt lõi từ project của bạn
from src.generative_model import call_model, conversation_memory
from src.config import initialize_once, search_index

# Khởi tạo 1 lần duy nhất khi API khởi động
initialize_once()

# Khởi tạo app FastAPI
app = FastAPI(
    title="RAG Chatbot API",
    description="API cho chatbot y tế sử dụng RAG"
)

# === CẤU HÌNH CORS ===
# Thêm dòng này để cho phép website (localhost:3000) của bạn gọi API này
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Sửa lại cổng nếu website của bạn chạy ở cổng khác
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức (GET, POST...)
    allow_headers=["*"],  # Cho phép tất cả các header
)


# === ĐỊNH NGHĨA INPUT/OUTPUT ===

class ChatRequest(BaseModel):
    """Mô hình dữ liệu cho request gửi đến"""
    question: str
    session_id: str = "default_api_session"  # Quản lý session nếu cần


class ChatResponse(BaseModel):
    """Mô hình dữ liệu cho response trả về"""
    answer: str
    session_id: str


# === TẠO ENDPOINT (API) ===

@app.get("/")
def read_root():
    return {"message": "Chào mừng đến với RAG Chatbot API!"}


@app.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    """
    Endpoint chính để xử lý chat
    """
    print(f"[API Request] Nhận câu hỏi: {request.question} | Session: {request.session_id}")

    # === Logic RAG (giống hệt main.py) ===

    question = request.question
    session_id = request.session_id

    # 1. Tạo query có ngữ cảnh
    chat_history = conversation_memory.chat_memory.messages[-4:]
    contextual_query = question

    if chat_history:
        history_str = "\n".join([
            f"{'User' if msg.type == 'human' else 'Bot'}: {msg.content[:100]}..."
            for msg in chat_history
        ])
        contextual_query = f"Lịch sử:\n{history_str}\n\nCâu hỏi mới: {question}"
        print(f"[Debug] Đang dùng query có ngữ cảnh để tìm kiếm...")
    else:
        print(f"[Debug] Lịch sử rỗng, dùng câu hỏi gốc để tìm kiếm.")

    # 2. Tìm kiếm (RAG)
    docs, similarity_score = search_index(contextual_query)

    # 3. Gọi model
    # (Đảm bảo hàm call_model của bạn trả về 'full_response')
    answer = call_model(
        question=question,
        docs=docs,
        session_id=session_id,
        similarity_score=similarity_score
    )

    print(f"[API Response] Trả lời: {answer[:50]}...")

    # 4. Trả về kết quả
    return ChatResponse(answer=answer, session_id=session_id)


# === CHẠY API ===
if __name__ == "__main__":
    # Chạy API server trên cổng 8000
    print("Khởi động API server tại http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)