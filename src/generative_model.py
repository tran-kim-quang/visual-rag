import google.generativeai as genai
from dotenv import load_dotenv
import os

# Imports cho phiên bản LangChain mới (1.0+)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from .config import search_index

load_dotenv()

# ============================================================ 
# FINETUNED MODEL CONFIG (COMMENTED)
# ============================================================ 
# # Load checkpoint path với absolute path
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# checkpoint_path = os.getenv('CHECKPOINT', 'results/checkpoint-2200')
# 
# # Convert to absolute path if relative
# if not os.path.isabs(checkpoint_path):
#     ADAPTER_PATH = os.path.join(BASE_DIR, checkpoint_path)
# else:
#     ADAPTER_PATH = checkpoint_path
# 
BASE_MODEL_NAME = os.getenv('BASE_MODEL_NAME')

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
client = genai.GenerativeModel(BASE_MODEL_NAME)

# Tạo custom memory classes để thay thế ConversationBufferMemory
class SimpleConversationMemory:
    def __init__(self, memory_key="chat_history", return_messages=True, max_token_limit=2000):
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.max_token_limit = max_token_limit
        self.chat_memory = InMemoryChatMessageHistory()

class SimpleVectorMemory:
    def __init__(self, retriever, memory_key="conv_vectors", return_source_documents=True):
        self.retriever = retriever
        self.memory_key = memory_key
        self.return_source_documents = return_source_documents

# Setup langchain
embeddings = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
chroma_store = Chroma(embedding_function=embeddings, persist_directory="chroma_conversations")

# Tạo retriever từ vectorstore
retriever = chroma_store.as_retriever(search_kwargs={"k": 3})

# Memory chính để lưu toàn bộ lịch sử (sử dụng custom class)
conversation_memory = SimpleConversationMemory(
    memory_key="chat_history",
    return_messages=True,
    max_token_limit=2000
)

# Vector memory để tìm kiếm semantic trong lịch sử (sử dụng custom class)
vector_memory = SimpleVectorMemory(
    retriever=retriever,
    memory_key="conv_vectors",
    return_source_documents=True
)
# Ngưỡng similarity để quyết định có phải câu hỏi y tế không
# Điều chỉnh dựa trên kết quả thực tế: 0.3-0.4 là câu hỏi y tế
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.3'))


def classify_by_similarity(similarity_score: float) -> str:
    if similarity_score >= SIMILARITY_THRESHOLD:
        return 'medical_question'
    else:
        return 'chitchat'


# ============================================================ 
# Bỏ filter keyword - để model tự handle
# ============================================================ 

def summarize_conver_context(question: str, docs: list, session_id: str, use_medical_docs: bool = True) -> tuple:
    medical_context = ""
    if use_medical_docs and docs:
        print(f"[Debug] Nhận được {len(docs)} documents từ search_index")

        # Giữ threshold 0.5 (hoặc 0.4) là tốt
        filtered_docs = [d for d in docs if d.get('similarity_score', 0) >= 0.4]

        if not filtered_docs:
            print("[Warning] Không có document nào đạt similarity >= 0.4")
            return "", ""

        top_docs = filtered_docs[:2]
        medical_parts = []

        for i, d in enumerate(top_docs):
            if not d.get("content") or len(d.get("content", "").strip()) < 10:
                print(f"[Warning] Document {i} có nội dung không hợp lệ: {d}")
                continue

            # === FIX: Tăng giới hạn cắt bớt context ===
            # Tăng từ 400 lên 1500 để tránh lỗi "thông tin bị cắt cụt"
            content = d["content"][:1500] + "..." if len(d["content"]) > 1500 else d["content"]

            if len(content.strip()) > 10:
                medical_parts.append(f"[{d.get('title', 'N/A')}]: {content}")
                print(f"[Debug] Sử dụng document: {d.get('title', 'N/A')[:50]}...")
            else:
                print(f"[Warning] Bỏ qua document có nội dung quá ngắn: {content[:50]}...")

        medical_context = "\n\n".join(medical_parts) if medical_parts else ""
        print(f"[Debug] Medical context length: {len(medical_context)} characters")

    conversation_history = ""
    try:
        chat_history = conversation_memory.chat_memory.messages[-6:] if len(
            conversation_memory.chat_memory.messages) > 0 else []
        if chat_history:
            conv_parts = []
            for msg in chat_history:
                role = "Người dùng" if msg.type == "human" else "Trợ lý"
                content = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                conv_parts.append(f"{role}: {content}")
            conversation_history = "\n".join(conv_parts)
    except:
        pass

    return medical_context, conversation_history


def call_model(question: str, docs: list, session_id: str, similarity_score: float = 0.0) -> str:
    intent = classify_by_similarity(similarity_score)
    print(f"[Intent: {intent} | Similarity: {similarity_score:.3f}]")

    use_medical_docs = (intent == 'medical_question' and similarity_score >= 0.4)

    # SỬA 1: Dùng 'question' (duy nhất)
    medical_context, conversation_history = summarize_conver_context(question, docs, session_id, use_medical_docs)

    # === FIX 1: Xóa yêu cầu trả lời ngắn ===
    if intent == 'medical_question':
        system_prompt = '''Bạn là trợ lý y tế AI. Trả lời chính xác dựa vào tài liệu y tế được cung cấp.
QUAN TRỌNG:
- Chỉ trả lời dựa vào thông tin y tế được cung cấp
- Không bịa đặt thông tin
- Nếu không có thông tin → nói "Tôi không có đủ thông tin"'''
    else:
        system_prompt = '''Bạn là trợ lý AI thân thiện.'''
    # ======================================

    history = []

    try:
        total_history = len(conversation_memory.chat_memory.messages)
        print(f"[Debug] Tổng lịch sử trong conversation memory: {total_history} messages")

        if total_history > 0:
            relevant_messages = []

            num_recent = min(3, total_history)
            recent_messages = conversation_memory.chat_memory.messages[-num_recent:]
            relevant_messages.extend(recent_messages)
            print(f"[Debug] Lấy {num_recent} messages gần nhất (recent context)")

            try:
                # SỬA 2: Dùng 'question'
                relevant_docs = vector_memory.retriever.vectorstore.similarity_search(
                    question,
                    k=5,
                    filter={"session_id": session_id}
                )

                print(f"[Debug] Tìm thấy {len(relevant_docs)} messages liên quan bằng semantic search")

                for doc in relevant_docs:
                    found = False
                    for msg in conversation_memory.chat_memory.messages:
                        if msg.content == doc.page_content and msg not in relevant_messages:
                            relevant_messages.append(msg)
                            found = True
                            print(f"[Debug] Added relevant: {msg.content[:50]}...")
                            break

                    if not found:
                        print(f"[Debug] Không tìm thấy message thực tế cho doc: {doc.page_content[:50]}...")
                        print(f"[Debug] Doc này có thể từ search khác hoặc không match chính xác")

            except Exception as e:
                print(f"[Debug] Không thể semantic search: {e}, dùng fallback keyword")

                all_messages = conversation_memory.chat_memory.messages
                for msg in all_messages[:-num_recent]:
                    msg_lower = msg.content.lower()
                    # SỬA 3: Dùng 'question'
                    question_lower = question.lower()

                    if 'tên' in question_lower or 'gì' in question_lower:
                        keywords = ['tên', 'là', 'gọi']
                    elif any(word in question_lower for word in ['đau', 'bệnh', 'triệu chứng']):
                        keywords = ['đau', 'bị', 'triệu chứng', 'cảm thấy']
                    else:
                        keywords = ['tôi', 'mình', 'bạn']

                    for keyword in keywords:
                        if keyword in question_lower and keyword in msg_lower:
                            if msg not in relevant_messages:
                                relevant_messages.append(msg)
                                print(f"[Debug] Added keyword relevant: {msg.content[:50]}...")
                                break

            if len(relevant_messages) > 8:
                relevant_messages.sort(key=lambda x: conversation_memory.chat_memory.messages.index(
                    x) if x in conversation_memory.chat_memory.messages else 999)
                relevant_messages = relevant_messages[-8:]

            for msg in relevant_messages:
                role = "user" if msg.type == "human" else "model"
                content = msg.content[:250] if len(msg.content) > 250 else msg.content
                history.append({"role": role, "parts": [{"text": content}]})

    except Exception as e:
        print(f"[Warning] Không thể load lịch sử từ conversation memory: {e}")

    # SỬA 4: Dùng 'question'
    if intent == 'medical_question' and medical_context and len(medical_context.strip()) > 50:
        current_msg = f'''Tài liệu y tế:
{medical_context}

Dựa vào tài liệu trên, trả lời câu hỏi: {question}'''
    else:
        if intent == 'medical_question':
            current_msg = f"Câu hỏi: {question}\n(Không tìm thấy tài liệu liên quan, trả lời theo kiến thức chung)"
        else:
            current_msg = question

    model = genai.GenerativeModel(
        BASE_MODEL_NAME,
        system_instruction=system_prompt
    )

    chat_session = model.start_chat(
        history=history
    )

    print("\n[Đang tạo câu trả lời...]")
    print(f"[Debug] Số messages: {len(history)}, Intent: {intent}")

    if intent == 'chitchat':
        temperature = 0.7
        max_tokens = 150
        stop_sequences = ["\n\n"]
    else:
        temperature = 0.3
        max_tokens = 2048  # SỬA LỖI CẮT CỤT: Tăng max_tokens
        stop_sequences = ["Tài liệu:"]

    # === FIX 2: Tắt stream và sửa lỗi crash ===
    try:
        print("[Debug] Bắt đầu gửi yêu cầu (non-stream)...")
        response = chat_session.send_message(
            current_msg,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                stop_sequences=stop_sequences,
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
            ),
            safety_settings= {
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
            },
            stream=False  # Đổi thành False
        )

        full_response = ""
        try:
            # Lấy thẳng response.text thay vì lặp stream
            full_response = response.text
            print(full_response)
        except ValueError:
            # Xử lý khi response bị chặn (ví dụ: safety block)
            print(f"[Warning] Model bị chặn (safety settings). Response: {response.prompt_feedback}")
            full_response = "Tôi không thể trả lời câu hỏi này do bộ lọc an toàn."
        except Exception as e_text:
            print(f"[Warning] Không thể đọc response.text: {e_text}")
            full_response = "Xin lỗi, tôi gặp lỗi khi đọc phản hồi."


        print(f"\n[Debug] Response length: {len(full_response)} chars")

        if not full_response.strip():
            print("[Warning] Model trả về rỗng!")
            full_response = "Xin lỗi, tôi không thể trả lời câu hỏi này lúc này."

    except Exception as e:
        print(f"\n[Lỗi khi gọi Google AI]: {e}")
        full_response = "Xin lỗi, tôi không thể xử lý câu hỏi của bạn lúc này."
    # ==========================================

    # SỬA 5: Dùng 'question' (duy nhất)
    save_conversation(question, full_response, session_id)

    return full_response
def save_conversation(question: str, answer: str, session_id: str) -> None:
    # Lưu vào conversation_memory (in-memory, để conversation context)
    conversation_memory.chat_memory.add_message(HumanMessage(content=question))
    conversation_memory.chat_memory.add_message(AIMessage(content=answer))
    print(f"[Debug] Đã lưu vào conversation memory. Total: {len(conversation_memory.chat_memory.messages)} messages")

    # Lưu vào vector_memory (persistent, để /history command và semantic search)
    question_doc = Document(
        page_content=question,
        metadata={"role": "user", "session_id": session_id}
    )

    answer_doc = Document(
        page_content=answer,
        metadata={"role": "assistant", "session_id": session_id}
    )

    vector_memory.retriever.vectorstore.add_documents([question_doc, answer_doc])


def get_conversation_history(session_id: str, limit: int = 10) -> list:
    """Lấy lịch sử cuộc trò chuyện theo session_id"""
    results = vector_memory.retriever.vectorstore.similarity_search(
        "",  # Query rỗng để lấy tất cả
        k=limit * 2,  # x2 vì có cả user và assistant
        filter={"session_id": session_id}
    )

    # Sắp xếp và format
    conversations = []
    for doc in results:
        conversations.append({
            "role": doc.metadata.get("role", "unknown"),
            "content": doc.page_content,
            "session_id": doc.metadata.get("session_id")
        })

    return conversations


def print_conversation_history(session_id: str, limit: int = 10) -> None:
    history = get_conversation_history(session_id, limit)

    if not history:
        print(f"Không tìm thấy lịch sử cho session: {session_id}")
        return

    print(f"\n{'=' * 60}")
    print(f"LỊCH SỬ CUỘC TRÒ CHUYỆN - Session: {session_id}")
    print(f"{ '=' * 60}\n")

    for i, conv in enumerate(history, 1):
        role_display = "[NGƯỜI DÙNG]" if conv["role"] == "user" else "[TRỢ LÝ]"
        print(f"{i}. {role_display}:")
        print(f"   {conv['content'][:200]}{'...' if len(conv['content']) > 200 else ''}")
        print()