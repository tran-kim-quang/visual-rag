# Comment: Cách cũ sử dụng transformers và HuggingFace models
from transformers import TextStreamer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from dotenv import load_dotenv
import os
# from openai import OpenAI

# Sửa imports - sử dụng langchain_community
from langchain.memory import ConversationBufferMemory, VectorStoreRetrieverMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import search_index

load_dotenv()

# Comment: Không cần load checkpoint khi dùng Ollama
relative_path = os.getenv('CHECKPOINT')
absolute_path = os.path.abspath(relative_path)

BASE_MODEL_NAME = os.getenv('BASE_MODEL_NAME')
ADAPTER_PATH = absolute_path

# Khởi tạo OpenAI client cho Ollama
# client = OpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key="ollama"  # Ollama không cần API key thật, chỉ cần placeholder
# )

# Setup langchain
embeddings = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
chroma_store = Chroma(embedding_function=embeddings, persist_directory="chroma_conversations")

# Tạo retriever từ vectorstore
retriever = chroma_store.as_retriever(search_kwargs={"k": 3})

vector_memory = VectorStoreRetrieverMemory(
    retriever=retriever,  # Sử dụng retriever
    memory_key="conv_vectors",
    return_source_documents=False
)

buffer_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# Ngưỡng similarity để quyết định có phải câu hỏi y tế không
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))

def classify_by_similarity(similarity_score: float) -> str:
    """
    Phân loại intent dựa trên similarity score
    Chỉ có 2 loại:
    - medical_question: Hỏi về tài liệu y tế
    - chitchat: Nói chuyện bình thường
    """
    if similarity_score >= SIMILARITY_THRESHOLD:
        return 'medical_question'
    else:
        return 'chitchat'

def summarize_conver_context(question: str, docs: list, session_id: str, use_medical_docs: bool = True) -> str:
    """
    Tổng hợp context từ tài liệu y tế và lịch sử hội thoại
    
    Args:
        question: Câu hỏi
        docs: Tài liệu y tế được retrieval
        session_id: ID phiên trò chuyện
        use_medical_docs: Có sử dụng tài liệu y tế không
    """
    # Lấy lịch sử hội thoại
    conv_docs = vector_memory.retriever.vectorstore.similarity_search(
        question, k=3, filter={"session_id": session_id}
    )
    
    combind = []
    
    # Thêm tài liệu y tế nếu cần
    if use_medical_docs and docs:
        for d in docs:
            combind.append(d["content"])
    
    # Luôn thêm lịch sử hội thoại (để model hiểu ngữ cảnh)
    if conv_docs:
        combind.append("--- LỊCH SỬ HỘI THOẠI ---")
        for msg in conv_docs:
            combind.append(f"[{msg.metadata['role']}] {msg.page_content}")
    
    context = "\n\n".join(combind)
    return context


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

# def call_model(question: str, docs: list, session_id: str, similarity_score: float = 0.0) -> str:
#     # Phân loại intent: chỉ 2 loại
#     intent = classify_by_similarity(similarity_score)
#     print(f"[Intent: {intent} | Similarity: {similarity_score:.3f}]")
    
#     # Quyết định có dùng tài liệu y tế không
#     use_medical_docs = (intent == 'medical_question')
    
#     # Tổng hợp context
#     context = summarize_conver_context(question, docs, session_id, use_medical_docs)
    
#     # Prompt dựa trên intent
#     if intent == 'medical_question':
#         # Hỏi về tài liệu y tế
#         system_prompt = "Bạn là trợ lý y tế chuyên nghiệp. Hãy sử dụng ngữ cảnh được cung cấp để trả lời câu hỏi của người dùng một cách tóm tắt, chi tiết và chính xác."
#         user_prompt = f"""Dựa vào ngữ cảnh sau đây để trả lời câu hỏi ở cuối.
# Nếu ngữ cảnh không chứa thông tin, hãy trả lời là "Tôi không tìm thấy thông tin về vấn đề này trong tài liệu."

# --- NGỮ CẢNH ---
# {context}
# --- HẾT NGỮ CẢNH ---

# CÂU HỎI: {question}"""
    
#     else:  # chitchat
#         # Nói chuyện bình thường
#         system_prompt = "Bạn là trợ lý y tế thân thiện. Hãy trả lời một cách lịch sự và tự nhiên."
#         if context:
#             user_prompt = f"""Lịch sử hội thoại:
# {context}

# Câu nói của người dùng: {question}"""
#         else:
#             user_prompt = question

    # # Gọi Ollama qua OpenAI API
    # response = client.chat.completions.create(
    #     model=BASE_MODEL_NAME,
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt}
    #     ],
    #     # temperature=0.7,
    #     # max_tokens=512,
    #     stream=True
    # )
    
    # # Stream response và lưu lại câu trả lời
    # print("\nTrả lời:")
    # full_response = ""
    # for chunk in response:
    #     if chunk.choices[0].delta.content:
    #         content = chunk.choices[0].delta.content
    #         print(content, end="", flush=True)
    #         full_response += content
    # print("\n")
    
    # Lưu lịch sử cuộc trò chuyện vào vector memory
    # save_conversation(question, full_response, session_id)
    
    # return full_response

def save_conversation(question: str, answer: str, session_id: str) -> None:
    from langchain.schema import Document
    
    # Lưu câu hỏi
    question_doc = Document(
        page_content=question,
        metadata={"role": "user", "session_id": session_id}
    )
    
    # Lưu câu trả lời
    answer_doc = Document(
        page_content=answer,
        metadata={"role": "assistant", "session_id": session_id}
    )

    vector_memory.retriever.vectorstore.add_documents([question_doc, answer_doc])
    buffer_memory.chat_memory.add_user_message(question)
    buffer_memory.chat_memory.add_ai_message(answer)

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

def call_model(question: str,
               docs: list,
               session_id: str,
               similarity_score: float,  # Sửa: Thêm similarity_score vào đây
               finetuned_model, 
               tokenizer) -> None:
    
    intent = classify_by_similarity(similarity_score) # Giờ đã hợp lệ
    print(f"[Intent: {intent} | Similarity: {similarity_score:.3f}]")
    use_medical_docs = (intent == 'medical_question')

    # Lấy context cho prompt
    context = summarize_conver_context(question, docs, session_id, use_medical_docs)
    
    if intent == 'medical_question':
        system_prompt = "Bạn là trợ lý y tế chuyên nghiệp..." # (giữ nguyên)
        user_prompt = f"""Dựa vào ngữ cảnh sau đây để trả lời câu hỏi ở cuối.
Nếu ngữ cảnh không chứa thông tin, hãy trả lời là "Tôi không tìm thấy thông tin về vấn đề này trong tài liệu."

--- NGỮ CẢNH ---
{context}
--- HẾT NGỮ CẢNH ---

CÂU HỎI: {question}"""
    
    else: # chitchat
        system_prompt = "Bạn là trợ lý y tế thân thiện..." # (giữ nguyên)
        if context:
            user_prompt = f"""Lịch sử hội thoại:
{context}

Câu nói của người dùng: {question}"""
        else:
            user_prompt = question
            
    # NOTE: Hiện tại bạn chưa dùng system_prompt, có thể tích hợp sau nếu muốn
    inputs = tokenizer(user_prompt, return_tensors="pt").to("cuda")

    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True 
    )
    
    print("\nTrả lời:")
    # Sửa: `generate` sẽ stream trực tiếp ra console
    # `outputs` là một tensor chứa token IDs
    outputs = finetuned_model.generate(
        **inputs,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True, 
        temperature=0.7,
        top_p=0.9,
        streamer=streamer
    )
    
    # Sửa: Cách đúng để lấy lại chuỗi text từ output tensor
    # outputs[0] vì batch size là 1
    # inputs.input_ids.shape[1] để bỏ qua phần prompt đã nhập
    output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # In lại toàn bộ câu trả lời (streamer đã in rồi, dòng này có thể bỏ nếu muốn)
    # print(f"\n--- Full Response Captured ---\n{output_text}\n------------------------------")
    
    # Lưu lịch sử cuộc trò chuyện vào vector memory
    save_conversation(question, output_text, session_id)

if __name__ == "__main__":
    # Sửa: Tải model và tokenizer ngay từ đầu
    print("Đang tải model và tokenizer...")
    finetuned_model, tokenizer = load_model()
    print("Model đã sẵn sàng!")
    
    session_id = "meoconlonton" # Hoặc tạo session ID ngẫu nhiên
    
    while True:
        question = str(input("\nAsk me anything (gõ 'exit' để thoát): "))
        if question.lower() == 'exit':
            break
            
        # Luôn retrieval để lấy similarity score
        docs, similarity_score = search_index(question)
        
        # Sửa: Truyền đúng các tham số vào hàm call_model
        call_model(
            question=question, 
            docs=docs, 
            session_id=session_id, 
            similarity_score=similarity_score,
            finetuned_model=finetuned_model,
            tokenizer=tokenizer
        )