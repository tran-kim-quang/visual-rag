#visual/api.py
"""
RAG Chatbot API - Main Application
Provides endpoints for medical chatbot with RAG functionality
"""

from dotenv import load_dotenv
load_dotenv()

import uvicorn
import asyncio
import json
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.conversation_manager import ConversationManager
from src.generative_model import call_model
from src.config import initialize_once, search_index
from src.query_rewriter import rewrite_query
from src.data_processor import MedicalDataProcessor

# Initialize components
initialize_once()
processor = MedicalDataProcessor()

# FastAPI app setup
app = FastAPI(
    title="RAG Chatbot API",
    description="API cho chatbot y tế sử dụng RAG"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Conversation manager and semaphore
manager = ConversationManager(max_history=10)
semaphore = asyncio.Semaphore(10)


# Request models
class ChatRequest(BaseModel):
    question: str
    session_id: str = "default_api_session"


# Routes
@app.get("/")
def read_root():
    """Root endpoint - Health check"""
    return {"message": "Chào mừng đến với RAG Chatbot API!"}


@app.post("/chat")
async def handle_chat(request: ChatRequest):
    """
    Main chat endpoint - Handles user questions with RAG
    Supports streaming responses via SSE
    """
    async with semaphore:
        print(f"\n[API Request] Nhận câu hỏi: {request.question} | Session: {request.session_id}")

        question = request.question
        session_id = request.session_id
        
        # Validate input
        if not question or not question.strip():
            async def error_stream():
                yield f"data: {json.dumps({'content': 'Vui lòng nhập câu hỏi.'})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream")
        
        # Preprocess query
        question = processor.preprocess_query(question)
        if not question:
            async def error_stream():
                yield f"data: {json.dumps({'content': 'Câu hỏi không hợp lệ.'})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream")

        # Get chat history
        chat_history_dicts = manager.get_history(session_id)
        print(f"[Memory] Lấy lịch sử cho session: {session_id} (có {len(chat_history_dicts)} tin nhắn)")
        
        recent_history_dicts = chat_history_dicts[-4:] if chat_history_dicts else []

        # Query rewriting with context
        if len(recent_history_dicts) > 0:
            rewritten_query = rewrite_query(question, recent_history_dicts)
            print(f"[Query Rewrite] Original: {question}")
            print(f"[Query Rewrite] Rewritten: {rewritten_query}")
            
            if not rewritten_query or len(rewritten_query) < 3:
                rewritten_query = question
        else:
            rewritten_query = question
            print(f"[Debug] Lịch sử rỗng, dùng câu hỏi gốc")

        # Search documents
        docs, similarity_score = search_index(rewritten_query, k=15)
        
        # Fallback to original query if needed
        if not docs:
            print(f"[Fallback] Không tìm được docs, thử query gốc...")
            docs, similarity_score = search_index(question, k=15)
        elif similarity_score < 0.3:
            print(f"[Fallback] Rerank score thấp ({similarity_score:.2f}), thử query gốc...")
            docs_fallback, sim_fallback = search_index(question, k=15)
            if sim_fallback > similarity_score:
                docs = docs_fallback
                similarity_score = sim_fallback
                print(f"[Fallback] Dùng kết quả fallback (score: {sim_fallback:.2f})")
        
        print(f"[Debug] Tìm thấy {len(docs)} tài liệu với độ tương đồng: {similarity_score:.2f}")

        # Handle no documents found
        if not docs:
            async def no_docs_stream():
                yield f"data: {json.dumps({'content': 'Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu.'})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(no_docs_stream(), media_type="text/event-stream")
        
        # Stream response
        async def response_stream():
            full_answer = ""
            try:
                for chunk in call_model(question, docs, session_id, similarity_score):
                    full_answer += chunk
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                
                # Save to history
                manager.add_to_history(session_id, question, full_answer)
                print(f"[Memory] Đã lưu Q&A vào lịch sử của session {session_id}")
                
                # Signal end of stream
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                print(f"[Error] Lỗi khi streaming: {str(e)}")
                yield f"data: {json.dumps({'error': 'Đã xảy ra lỗi khi xử lý câu trả lời.'})}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(response_stream(), media_type="text/event-stream")


@app.get("/sessions")
def list_sessions():
    """List all active sessions"""
    return {
        "total_sessions": len(manager.history),
        "session_ids": list(manager.history.keys())
    }


@app.get("/session/{session_id}")
def get_session_history(session_id: str):
    """Get chat history for a specific session"""
    if not manager.has_history(session_id):
        return {"message": f"Session {session_id} không tồn tại hoặc rỗng."}
    
    history = manager.get_history(session_id)
    return {
        "session_id": session_id,
        "total_messages": len(history),
        "history": history
    }


@app.delete("/session/{session_id}")
def clear_session_endpoint(session_id: str):
    """Clear chat history for a specific session"""
    if manager.has_history(session_id):
        manager.clear_session(session_id)
        return {"message": f"Đã xóa session {session_id}"}
    return {"message": f"Session {session_id} không tồn tại"}


if __name__ == "__main__":
    print("Khởi động API server tại http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)