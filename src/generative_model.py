from openai import OpenAI

from dotenv import load_dotenv
import os

# Sửa imports - sử dụng langchain_community
from langchain.memory import ConversationBufferMemory, VectorStoreRetrieverMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import search_index

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

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'  # required nhưng không cần thật
)

# Setup langchain
embeddings = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
chroma_store = Chroma(embedding_function=embeddings, persist_directory="chroma_conversations")

# Tạo retriever từ vectorstore
retriever = chroma_store.as_retriever(search_kwargs={"k": 3})

# Memory chính để lưu toàn bộ lịch sử
conversation_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    max_token_limit=2000  # Giới hạn token để tránh quá dài
)

# Vector memory để tìm kiếm semantic trong lịch sử
vector_memory = VectorStoreRetrieverMemory(
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
    # PHẦN 1: Tài liệu y tế (chỉ lấy top 1-2, không cần quá nhiều)
    medical_context = ""
    if use_medical_docs and docs:
        # Kiểm tra tính toàn vẹn dữ liệu
        print(f"[Debug] Nhận được {len(docs)} documents từ search_index")

        # Chỉ lấy top 2 documents và giới hạn độ dài
        top_docs = docs[:2]
        medical_parts = []

        for i, d in enumerate(top_docs):
            # Kiểm tra document có đầy đủ thông tin không
            if not d.get("content") or len(d.get("content", "").strip()) < 10:
                print(f"[Warning] Document {i} có nội dung không hợp lệ: {d}")
                continue

            # Giới hạn mỗi doc tối đa 300 ký tự
            content = d["content"][:300] + "..." if len(d["content"]) > 300 else d["content"]

            # Đảm bảo content có ý nghĩa
            if len(content.strip()) > 10:
                medical_parts.append(f"[{d.get('title', 'N/A')}]: {content}")
                print(f"[Debug] Sử dụng document: {d.get('title', 'N/A')[:50]}...")
            else:
                print(f"[Warning] Bỏ qua document có nội dung quá ngắn: {content[:50]}...")

        medical_context = "\n\n".join(medical_parts) if medical_parts else ""
        print(f"[Debug] Medical context length: {len(medical_context)} characters")
    
    # PHẦN 2: Lịch sử hội thoại gần đây (chỉ lấy 2-3 turn gần nhất)
    conversation_history = ""
    try:
        # Lấy từ buffer_memory (đảm bảo thứ tự thời gian)
        chat_history = buffer_memory.chat_memory.messages[-4:] if len(buffer_memory.chat_memory.messages) > 0 else []
        if chat_history:
            conv_parts = []
            for msg in chat_history:
                role = "Người dùng" if msg.type == "human" else "Trợ lý"
                # Giới hạn mỗi message tối đa 100 ký tự
                content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                conv_parts.append(f"{role}: {content}")
            conversation_history = "\n".join(conv_parts)
    except:
        pass
    
    return medical_context, conversation_history

def call_model(question: str, docs: list, session_id: str, similarity_score: float = 0.0) -> str:
    # Phân loại intent: chỉ 2 loại
    intent = classify_by_similarity(similarity_score)
    print(f"[Intent: {intent} | Similarity: {similarity_score:.3f}]")
    
    # Quyết định có dùng tài liệu y tế không
    use_medical_docs = (intent == 'medical_question' and similarity_score >= 0.3)
    
    # Tổng hợp context (trả về tuple)
    medical_context, conversation_history = summarize_conver_context(question, docs, session_id, use_medical_docs)
    
    # Xây dựng messages với LỊCH SỬ HỘI THOẠI
    messages = []
    
    # System prompt RÕ RÀNG về vai trò
    if intent == 'medical_question':
        system_prompt = """Bạn là trợ lý y tế hỗ trợ người dùng. Trả lời ngắn gọn dựa vào lịch sử hội thoại và thông tin y tế."""
    else:
        system_prompt = """Bạn là trợ lý hỗ trợ người dùng. Trả lời ngắn gọn 1-2 câu."""
    
    messages.append({"role": "system", "content": system_prompt})
    
    # Thêm LỊCH SỬ HỘI THOẠI với SEMANTIC RETRIEVAL (như dữ liệu y tế!)
    try:
        total_history = len(conversation_memory.chat_memory.messages)
        print(f"[Debug] Tổng lịch sử trong conversation memory: {total_history} messages")

        if total_history > 0:
            # CHIẾN LƯỢC: SEMANTIC RETRIEVAL như dữ liệu y tế!
            # 1. Luôn lấy 2-3 messages gần nhất (recent context)
            # 2. Dùng vector_memory để tìm TOP K messages liên quan nhất bằng SEMANTIC SEARCH

            relevant_messages = []

            # Bước 1: Lấy recent context (2-3 messages gần nhất)
            num_recent = min(3, total_history)
            recent_messages = conversation_memory.chat_memory.messages[-num_recent:]
            relevant_messages.extend(recent_messages)
            print(f"[Debug] Lấy {num_recent} messages gần nhất (recent context)")

            # Bước 2: SEMANTIC SEARCH trong toàn bộ lịch sử (như retrieval dữ liệu y tế!)
            try:
                # Tìm TOP 5 messages liên quan nhất bằng semantic search
                relevant_docs = vector_memory.retriever.vectorstore.similarity_search(
                    question,  # Query giống như tìm kiếm dữ liệu y tế
                    k=5,  # Lấy top 5 liên quan nhất
                    filter={"session_id": session_id}
                )

                print(f"[Debug] Tìm thấy {len(relevant_docs)} messages liên quan bằng semantic search")

                # Thêm messages liên quan (tránh trùng lặp)
                for doc in relevant_docs:
                    # Tìm message tương ứng trong conversation memory
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
                        # Bỏ qua doc này thay vì tạo fake message

            except Exception as e:
                print(f"[Debug] Không thể semantic search: {e}, dùng fallback keyword")

                # Fallback: Keyword search trong toàn bộ lịch sử
                all_messages = conversation_memory.chat_memory.messages
                for msg in all_messages[:-num_recent]:  # Bỏ qua recent đã có
                    msg_lower = msg.content.lower()
                    question_lower = question.lower()

                    # Từ khóa quan trọng cho từng loại câu hỏi
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

            # Giới hạn tổng số messages để tránh quá dài
            if len(relevant_messages) > 8:
                # Sắp xếp theo thời gian (recent nhất trước)
                relevant_messages.sort(key=lambda x: conversation_memory.chat_memory.messages.index(x) if x in conversation_memory.chat_memory.messages else 999)
                relevant_messages = relevant_messages[-8:]  # Giữ 8 gần nhất

            # Thêm vào messages với format đúng cho Ollama
            for msg in relevant_messages:
                role = "user" if msg.type == "human" else "assistant"
                content = msg.content[:250] if len(msg.content) > 250 else msg.content
                messages.append({"role": role, "content": content})

    except Exception as e:
        print(f"[Warning] Không thể load lịch sử từ conversation memory: {e}")
    
    # Thêm câu hỏi hiện tại với context y tế (nếu có)
    if intent == 'medical_question' and medical_context:
        # Kiểm tra tính toàn vẹn của medical_context
        if medical_context and len(medical_context.strip()) > 10:
            current_msg = f"Thông tin y tế liên quan:\n{medical_context}\n\nCâu hỏi: {question}"
        else:
            print(f"[Warning] Medical context quá ngắn hoặc rỗng: {medical_context}")
            current_msg = f"Câu hỏi: {question}"
    else:
        current_msg = question

    messages.append({"role": "user", "content": current_msg})
    
    print("\n[Đang tạo câu trả lời...]")
    print(f"[Debug] Số messages: {len(messages)}, Intent: {intent}")
    
    # Thiết lập parameters dựa trên intent
    # Dùng stop sequences thay vì max_tokens để dừng tự nhiên hơn
    if intent == 'chitchat':
        temperature = 0.6
        max_tokens = 150  # Tăng lên nhưng dùng stop để kiểm soát
        stop_sequences = ["\n\n", "Người dùng:", "User:", "["]
    else:
        temperature = 0.7
        max_tokens = 400  # Tăng lên, dùng stop để kiểm soát
        stop_sequences = ["\n\n\n", "Người dùng:", "User:", "---"]
    
    try:
        stream = client.chat.completions.create(
            model=BASE_MODEL_NAME,
            messages=messages,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            stop=stop_sequences,
        )
        
        # Streaming output
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)
                full_response += content
        
        print()  # Newline sau khi stream xong
        
    except Exception as e:
        print(f"\n[Lỗi khi gọi Ollama]: {e}")
        full_response = "Xin lỗi, tôi không thể xử lý câu hỏi của bạn lúc này."
    
    # Lưu lịch sử cuộc trò chuyện vào vector memory
    save_conversation(question, full_response, session_id)
    
    return full_response

def save_conversation(question: str, answer: str, session_id: str) -> None:
    from langchain.schema import Document

    # Lưu vào conversation_memory (in-memory, để conversation context)
    conversation_memory.chat_memory.add_user_message(question)
    conversation_memory.chat_memory.add_ai_message(answer)
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
    
    print(f"\n{'='*60}")
    print(f"LỊCH SỬ CUỘC TRÒ CHUYỆN - Session: {session_id}")
    print(f"{'='*60}\n")
    
    for i, conv in enumerate(history, 1):
        role_display = "[NGƯỜI DÙNG]" if conv["role"] == "user" else "[TRỢ LÝ]"
        print(f"{i}. {role_display}:")
        print(f"   {conv['content'][:200]}{'...' if len(conv['content']) > 200 else ''}")
        print()

# def call_model(question: str,
#                 docs: list,
#                 session_id: str,
#                 finetuned_model, tokenizer) -> None:
#     # finetuned_model, tokenizer = load_model()
#     context = summarize_conver_context(question, docs, session_id)
#     system_prompt = "Bạn là trợ lý y tế chuyên nghiệp. Hãy sử dụng ngữ cảnh được cung cấp để trả lời câu hỏi của người dùng một cách chi tiết và chính xác."
#     prompt = f"""[INST] <<SYS>>
#         {system_prompt}
#         <</SYS>>
#         Dựa vào ngữ cảnh sau đây để trả lời câu hỏi ở cuối.
#         Nếu ngữ cảnh không chứa thông tin, hãy trả lời là "Tôi không tìm thấy thông tin về vấn đề này trong tài liệu."
#         --- NGỮ CẢNH ---
#         {context}
#         --- HẾT NGỮ CẢNH ---
#         CÂU HỎI: {question} [/INST]"""
#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#
#     streamer = TextStreamer(
#         tokenizer,
#         skip_prompt=True,        
#         skip_special_tokens=True 
#     )
#     outputs = finetuned_model.generate(
#         **inputs,
#         max_new_tokens=512,
#         eos_token_id=tokenizer.eos_token_id,
#         do_sample=True, 
#         temperature=0.7,
#         top_p=0.9,
#         streamer=streamer
#     )

if __name__ == "__main__":
    session_id = "meoconlonton"
    question = str(input("Ask me anything: "))
    
    # Luôn retrieval để lấy similarity score
    docs, similarity_score = search_index(question)
    
    # Gọi model với similarity score
    call_model(question, docs, session_id, similarity_score)