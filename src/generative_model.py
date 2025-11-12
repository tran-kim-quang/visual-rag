from openai import OpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
import os

# Hallucination Blacklist
HALLUCINATION_BLACKLIST = [
    "lá tre", "nano bot", "chip điện tử", "chip vaccine",
    "nước chanh chữa ung thư", "tỏi chữa", "vitamin c chữa cảm",
    "thuốc nam chữa ung thư", "bài thuốc gia truyền chữa", 
    "công nghệ nano trong thuốc", "5g gây", "thuốc đông y chữa khỏi"
]

def check_hallucination(question: str) -> bool:
    """Kiểm tra câu hỏi có chứa từ khóa giả khoa học không"""
    q_lower = question.lower()
    return any(keyword in q_lower for keyword in HALLUCINATION_BLACKLIST)

# Global model and memory variables
client = None
conversation_memory = None
cross_encoder_reranker = None
BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "meta-llama/llama-3.1-8b-instruct:free")


def initialize_openai_client():
    """Khởi tạo và trả về Client OpenRouter (chuẩn OpenAI)."""
    global client
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY không được cấu hình")
            
        client = OpenAI(
          base_url="https://openrouter.ai/api/v1",
          api_key=api_key,
          default_headers={
              "HTTP-Referer": "http://localhost:8000",
              "X-Title": "Medical RAG Chatbot"
          }
        )
        print("[Debug] Đã khởi tạo OpenAI Client cho OpenRouter.")
        return client
    except Exception as e:
        print(f"[LỖI OpenRouter] Không thể khởi tạo Client: {e}")
        raise


def initialize_memory(session_id):
    """Khởi tạo bộ nhớ cuộc trò chuyện."""
    global conversation_memory
    conversation_memory = InMemoryChatMessageHistory()


def format_chat_history_for_openai(chat_history: list) -> list:
    """Hàm phụ: Chuyển đổi list [HumanMessage, AIMessage] sang list[dict]."""
    messages = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})
    return messages


def call_model(question, docs, session_id, similarity_score):
    """
    Hàm gọi mô hình chính cho chat tương tác, sử dụng conversation memory.
    """
    global conversation_memory, client
    
    try:
        # Check hallucination
        if check_hallucination(question):
            yield "Xin lỗi, tôi không thể tư vấn về phương pháp điều trị không có căn cứ khoa học hoặc thông tin sai lệch. Vui lòng hỏi về các phương pháp y khoa được công nhận."
            return
        
        # Initialize memory
        if conversation_memory is None:
            initialize_memory(session_id)
            
        # Initialize client
        if client is None:
            client = initialize_openai_client()
        
        # Get chat history
        chat_history = conversation_memory.messages[-4:] if conversation_memory.messages else []
        
        # Build context - Chỉ lấy nội dung relevant
        if not docs:
            yield "Xin lỗi, không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
            return
        
        # Lọc docs có rerank_score > 0.3
        relevant_docs = [d for d in docs if d.get('rerank_score', 0) > 0.3]
        if not relevant_docs:
            relevant_docs = docs[:3]  # Fallback: lấy top 3
            
        context = "=== THÔNG TIN Y KHOA ===\n\n"
        for idx, doc in enumerate(relevant_docs[:5], 1):  # Chỉ lấy top 5
            content = doc.get('content', '') if isinstance(doc, dict) else str(doc)
            title = doc.get('title', 'N/A')
            context += f"[Nguồn {idx}: {title}]\n{content[:400]}\n\n"
        
        # System prompt - Cải thiện độ chính xác
        system_prompt = (
            "Bạn là trợ lý y tế AI chuyên nghiệp. Trả lời CHÍNH XÁC dựa trên CONTEXT.\n\n"
            "QUY TẮC BẮT BUỘC:\n"
            "1. CHỈ trả lời từ CONTEXT - KHÔNG bịa đặt\n"
            "2. Nếu CONTEXT không có thông tin → Trả lời: 'Dữ liệu không có thông tin về...'\n"
            "3. Từ chối câu hỏi về: phương pháp dân gian, thuyết âm mưu, thuốc không rõ nguồn gốc\n"
            "4. Trả lời ngắn gọn, có số liệu cụ thể nếu có trong CONTEXT\n"
            "5. KHÔNG đưa ra lời khuyên y tế cá nhân - chỉ cung cấp thông tin\n\n"
            "CẤU TRÚC TRẢ LỜI:\n"
            "- Định nghĩa/Triệu chứng (nếu có trong CONTEXT)\n"
            "- Nguyên nhân (nếu có)\n"
            "- Điều trị (nếu có)\n"
            "- Lưu ý: 'Thông tin chỉ mang tính tham khảo. Hãy tham khảo ý kiến bác sĩ.'"
        )

        # Build messages
        history_messages_list = format_chat_history_for_openai(chat_history)
        messages_payload = [{"role": "system", "content": system_prompt}]
        messages_payload.extend(history_messages_list)
        messages_payload.append({
            "role": "user", 
            "content": f"{context}\n\n**CÂU HỎI:** {question}\n\nTrả lời ngắn gọn, chính xác dựa trên THÔNG TIN Y KHOA ở trên."
        })

        # Stream response
        stream = client.chat.completions.create(
            model=BASE_MODEL_NAME,
            messages=messages_payload,
            temperature=0.0,  # Giảm xuống 0 để chính xác hơn
            max_tokens=800,   # Giảm xuống để ngắn gọn hơn
            top_p=0.9,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content
        
        # Save to memory
        if full_response:
            conversation_memory.add_user_message(HumanMessage(content=question))
            conversation_memory.add_ai_message(AIMessage(content=full_response))

    except Exception as e:
        error_msg = f"[ERROR call_model]: {str(e)}"
        print(error_msg)
        import traceback
        print(traceback.format_exc())
        yield "Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn."