# src/config.py
import json
import numpy as np
import hnswlib
import os
from .embed_data import EmbeddingProcessor
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.getenv("DATA_FILE") or os.path.join(BASE_DIR, "data_clean/medical_data_with_embeddings.json")
INDEX_FILE = os.getenv("INDEX_FILE") or os.path.join(BASE_DIR, "data_clean/medical_index.hnsw")
SUMMARIZE_MODEL = os.getenv("SUMMARIZE_MODEL")

if not os.path.isabs(DATA_FILE):
    DATA_FILE = os.path.join(BASE_DIR, DATA_FILE)
if not os.path.isabs(INDEX_FILE):
    INDEX_FILE = os.path.join(BASE_DIR, INDEX_FILE)

# Global variables
original_data = None
p_loaded = None
process = None
_initialized = False


def initialize_once():
    global _initialized, original_data, p_loaded, process

    if _initialized:
        return

    print(f"Đang tải dữ liệu gốc từ '{DATA_FILE}'...")
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        print(f"✓ Tải thành công {len(original_data)} chunks.")
    except Exception as e:
        print(f"[LỖI] Không thể tải {DATA_FILE}: {e}")
        return

    print(f"Đang tải index HNSW từ '{INDEX_FILE}'...")
    try:
        num_dimensions = len(original_data[0]['embedding'])
        p_loaded = hnswlib.Index(space='cosine', dim=num_dimensions)
        p_loaded.load_index(INDEX_FILE)
        p_loaded.set_ef(100)
        print(f"✓ Tải index HNSW thành công (Dimensions: {num_dimensions}).")
    except Exception as e:
        print(f"[LỖI] Không thể tải index: {e}")
        return

    print("Đang khởi tạo EmbeddingProcessor...")
    try:
        process = EmbeddingProcessor()
        print("✓ Khởi tạo EmbeddingProcessor thành công.\n")
    except Exception as e:
        print(f"[LỖI] Không thể khởi tạo EmbeddingProcessor: {e}")
        return

    _initialized = True


def check_index(data_path, index_path):
    print(f"Bắt đầu xây dựng index từ '{data_path}'...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    embeddings = np.array([item['embedding'] for item in data]).astype('float32')
    if embeddings.size == 0:
        print("Không tìm thấy embedding.")
        return None

    num_dimensions = embeddings.shape[1]
    num_elements = len(embeddings)

    p = hnswlib.Index(space='cosine', dim=num_dimensions)
    p.init_index(max_elements=num_elements, ef_construction=400, M=32)
    p.add_items(embeddings, np.arange(num_elements))
    p.save_index(index_path)
    print(f"✓ Index đã được xây dựng với {num_elements} vector và lưu tại '{index_path}'")
    return data


def search_index(query, k=10):
    if not p_loaded or not process or not original_data:
        print("[LỖI] Hệ thống RAG chưa sẵn sàng.")
        return [], 0.0

    query_vector = process.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

    try:
        labels, distances = p_loaded.knn_query(query_vector, k=k)
        similarities = 1 - distances[0]
        max_similarity = float(similarities[0]) if len(similarities) > 0 else 0.0

        print(f"\n--- {k} kết quả tìm kiếm hàng đầu ---")
        print(f"Similarity score cao nhất: {max_similarity:.3f}")

        results = []
        for i, label_id in enumerate(labels[0]):
            item = original_data[label_id]
            item['similarity_score'] = float(similarities[i])
            results.append(item)
            print(f"[{similarities[i]:.3f}] ID: {item.get('id', 'N/A')}, Title: {item.get('title', 'N/A')}")
            print(f"         Content: {item.get('content', '')[:50]}...")

        return results, max_similarity
    except Exception as e:
        print(f"[LỖI] Lỗi khi tìm kiếm: {e}")
        return [], 0.0