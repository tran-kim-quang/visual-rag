from generative_model import print_conversation_history, get_conversation_history
import sys

def main():
    # Lấy session_id từ command line hoặc dùng mặc định
    if len(sys.argv) > 1:
        session_id = sys.argv[1]
    else:
        session_id = input("Nhập session ID (mặc định: meoconlonton): ").strip()
        if not session_id:
            session_id = "meoconlonton"
    
    # Lấy số lượng tin nhắn muốn xem
    if len(sys.argv) > 2:
        limit = int(sys.argv[2])
    else:
        limit_input = input("Số lượng tin nhắn muốn xem (mặc định: 10): ").strip()
        limit = int(limit_input) if limit_input else 10
    
    # In lịch sử
    print_conversation_history(session_id, limit)
    
    # Tùy chọn: xuất ra file
    export = input("\nBạn có muốn xuất lịch sử ra file? (y/n): ").strip().lower()
    if export == 'y':
        history = get_conversation_history(session_id, limit)
        filename = f"conversation_history_{session_id}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"LỊCH SỬ CUỘC TRÒ CHUYỆN - Session: {session_id}\n")
            f.write("="*60 + "\n\n")
            
            for i, conv in enumerate(history, 1):
                role_display = "NGƯỜI DÙNG" if conv["role"] == "user" else "TRỢ LÝ"
                f.write(f"{i}. {role_display}:\n")
                f.write(f"{conv['content']}\n\n")
        
        print(f"Đã xuất lịch sử ra file: {filename}")

if __name__ == "__main__":
    main()

