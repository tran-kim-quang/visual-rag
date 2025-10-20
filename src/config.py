import json
import numpy as np
import hnswlib
# from openai import OpenAI
import os
# from src.embed_data import EmbeddingProcessor
from embed_data import EmbeddingProcessor
from dotenv import load_dotenv
load_dotenv()

# Get paths from .env or use defaults with absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.getenv("DATA_FILE") or os.path.join(BASE_DIR, "data_clean/data_with_embeddings.json")
INDEX_FILE = os.getenv("INDEX_FILE") or os.path.join(BASE_DIR, "data_clean/medical_index.hnsw")
SUMMARIZE_MODEL = os.getenv("SUMMARIZE_MODEL", "gemma2:2b")

# Convert to absolute path if relative
if not os.path.isabs(DATA_FILE):
    DATA_FILE = os.path.join(BASE_DIR, DATA_FILE)
if not os.path.isabs(INDEX_FILE):
    INDEX_FILE = os.path.join(BASE_DIR, INDEX_FILE)
def check_index(data_path, index_path):
    """Đọc dữ liệu, xây dựng và lưu index HNSW."""
    print("Loading...")
    if data_path:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    embeddings = np.array([item['embedding'] for item in data]).astype('float32')
    
    if embeddings.size == 0:
        print("Không tìm thấy embedding trong file dữ liệu.")
        return None, None

    num_dimensions = embeddings.shape[1]
    num_elements = len(embeddings)

    p = hnswlib.Index(space='cosine', dim=num_dimensions)
    # Tăng ef_construction và M để tìm kiếm chính xác hơn
    # ef_construction: cao hơn = chậm hơn nhưng chính xác hơn (200 -> 400)
    # M: số kết nối tối đa (16 -> 32)
    p.init_index(max_elements=num_elements, ef_construction=400, M=32)
    p.add_items(embeddings, np.arange(num_elements))
    
    p.save_index(index_path)
    print(f"Index đã được xây dựng với {num_elements} vector và lưu tại '{index_path}'")
    
    return data 

def search_index(query, index_path=INDEX_FILE, k=3):
    process = EmbeddingProcessor()
    query_vector = process.embed_query(query)

    if not os.path.exists(index_path):
        print(f"File index '{index_path}' không tồn tại.")
        return [], 0.0
    
    original_data = check_index(DATA_FILE, INDEX_FILE)
    # Lấy số chiều từ vector truy vấn
    num_dimensions = len(query_vector)

    p_loaded = hnswlib.Index(space='cosine', dim=num_dimensions)
    p_loaded.load_index(index_path)
    
    # Tăng ef để tìm kiếm chính xác hơn (mặc định là 10, tăng lên 100)
    # ef càng cao, tìm kiếm càng chính xác nhưng chậm hơn
    p_loaded.set_ef(100)

    labels, distances = p_loaded.knn_query(query_vector, k=k)
    
    # Convert cosine distance to cosine similarity: similarity = 1 - distance
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
