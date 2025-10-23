# build_index.py
import os
from src.config import check_index
from dotenv import load_dotenv

load_dotenv()

"""
Script này chỉ chạy MỘT LẦN để xây dựng HNSW index
từ file JSON chứa 34,667 chunks.
"""

# Lấy đường dẫn từ .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.getenv("DATA_FILE") or os.path.join(BASE_DIR, "data_clean/medical_data_with_embeddings.json")
INDEX_FILE = os.getenv("INDEX_FILE") or os.path.join(BASE_DIR, "data_clean/medical_index.hnsw")

# Chuyển đổi đường dẫn
if not os.path.isabs(DATA_FILE):
    DATA_FILE = os.path.join(BASE_DIR, DATA_FILE)
if not os.path.isabs(INDEX_FILE):
    INDEX_FILE = os.path.join(BASE_DIR, INDEX_FILE)

if __name__ == "__main__":
    print(f"Đang đọc từ: {DATA_FILE}")
    print(f"Sẽ lưu vào: {INDEX_FILE}")
    print("Bắt đầu xây dựng index...")

    # Gọi hàm build index từ file config của bạn
    check_index(DATA_FILE, INDEX_FILE)

    print("\n✓ Xây dựng index HNSW hoàn tất!")