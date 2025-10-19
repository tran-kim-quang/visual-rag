import json
import numpy as np
import hnswlib
from openai import OpenAI
import os
from src.embed_data import EmbeddingProcessor
from dotenv import load_dotenv
load_dotenv()
DATA_FILE = os.getenv("DATA_FILE")
INDEX_FILE = os.getenv("INDEX_FILE")
SUMMARIZE_MODEL = os.getenv("SUMMARIZE_MODEL")
def check_index(data_path, index_path):
    """Đọc dữ liệu, xây dựng và lưu index HNSW."""
    print("Loading...")
    if data_path:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    embeddings = np.array([item['embedding'] for item in data]).astype('float32')
    
    if embeddings.size == 0:
        print("Không tìm thấy embedding trong file dữ liệu.")
        return None, None

    num_dimensions = embeddings.shape[1]
    num_elements = len(embeddings)

    p = hnswlib.Index(space='ip', dim=num_dimensions)
    p.init_index(max_elements=num_elements, ef_construction=200, M=16)
    p.add_items(embeddings, np.arange(num_elements))
    
    p.save_index(index_path)
    print(f"Index đã được xây dựng với {num_elements} vector và lưu tại '{index_path}'")
    
    return data # Trả về data để tra cứu nội dung

def search_index(query, index_path=INDEX_FILE, k=3):
    """Tải index và thực hiện tìm kiếm."""
    process = EmbeddingProcessor()
    query_vector = process.embed_query(query)

    if not os.path.exists(index_path):
        print(f"File index '{index_path}' không tồn tại.")
        return
    original_data = check_index(DATA_FILE, INDEX_FILE)
    # Lấy số chiều từ vector truy vấn
    num_dimensions = len(query_vector)

    p_loaded = hnswlib.Index(space='ip', dim=num_dimensions)
    p_loaded.load_index(index_path)

    labels, distances = p_loaded.knn_query(query_vector, k=k)
    
    print(f"\\n--- {k} kết quả tìm kiếm hàng đầu ---")
    results = []
    for label_id in labels[0]:
        item = original_data[label_id]
        results.append(item)
        print(f"ID: {item.get('id', 'N/A')}, Title: {item.get('title', 'N/A')}")
        print(f"  Content: {item.get('content', '')[:50]}...")
    
    return results
