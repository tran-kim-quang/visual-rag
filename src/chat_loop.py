"""Script chat liên tục với user"""
from generative_model import call_model, print_conversation_history
from config import search_index
import sys

def main():
    # Lấy session_id
    if len(sys.argv) > 1:
        session_id = sys.argv[1]
    else:
        session_id = input("Nhập session ID của bạn (mặc định: meoconlonton): ").strip()
        if not session_id:
            session_id = "meoconlonton"
    
    print("\n" + "="*60)
    print(f"TRỢ LÝ Y TẾ - Session: {session_id}")
    print("="*60)
    print("\nCác lệnh đặc biệt:")
    print("  /history - Xem lịch sử hội thoại")
    print("  /clear   - Xóa màn hình")
    print("  /exit    - Thoát chương trình")
    print("\n" + "="*60 + "\n")
    
    while True:
        try:
            # Nhận input từ user
            question = input("\n[BẠN]: ").strip()
            
            if not question:
                continue
            
            # Xử lý các lệnh đặc biệt
            if question.lower() == '/exit':
                print("\nTạm biệt! Hẹn gặp lại.")
                break
            
            if question.lower() == '/history':
                print_conversation_history(session_id, limit=10)
                continue
            
            if question.lower() == '/clear':
                import os
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            
            # Retrieval để lấy similarity score
            print("[Đang phân tích câu hỏi...]")
            docs, similarity_score = search_index(question)
            
            # Gọi model để trả lời
            print("[TRỢ LÝ]:", end=" ")
            answer = call_model(question, docs, session_id, similarity_score)
            
        except KeyboardInterrupt:
            print("\n\nĐã dừng. Tạm biệt!")
            break
        except Exception as e:
            print(f"\n[LỖI]: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()

