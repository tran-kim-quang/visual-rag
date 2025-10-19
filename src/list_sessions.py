"""Script để liệt kê tất cả các session ID"""
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def list_all_sessions():
    """Liệt kê tất cả các session ID có trong database"""
    # Setup
    embeddings = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
    chroma_store = Chroma(embedding_function=embeddings, persist_directory="chroma_conversations")
    
    # Lấy tất cả documents
    try:
        # Thử lấy một số lượng lớn documents
        all_docs = chroma_store.similarity_search("", k=1000)
        
        # Lấy danh sách session_id duy nhất
        sessions = set()
        for doc in all_docs:
            session_id = doc.metadata.get("session_id")
            if session_id:
                sessions.add(session_id)
        
        if sessions:
            print("\n" + "="*60)
            print("DANH SÁCH CÁC SESSION")
            print("="*60 + "\n")
            
            for i, session in enumerate(sorted(sessions), 1):
                # Đếm số tin nhắn trong session
                session_docs = [d for d in all_docs if d.metadata.get("session_id") == session]
                count = len(session_docs)
                print(f"{i}. Session ID: {session}")
                print(f"   Số tin nhắn: {count}")
                print()
        else:
            print("Không tìm thấy session nào trong database.")
    
    except Exception as e:
        print(f"Lỗi khi truy vấn database: {e}")

if __name__ == "__main__":
    list_all_sessions()

