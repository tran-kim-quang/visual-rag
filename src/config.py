# src/config.py
import json
import numpy as np
import hnswlib
import os
import sys
from sentence_transformers import CrossEncoder
from .embed_data import EmbeddingProcessor
from dotenv import load_dotenv
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.getenv("DATA_FILE") or os.path.join(BASE_DIR, "data_clean/medical_data_with_embeddings.json")
INDEX_FILE = os.getenv("INDEX_FILE") or os.path.join(BASE_DIR, "data_clean/medical_index.hnsw")

if not os.path.isabs(DATA_FILE):
    DATA_FILE = os.path.join(BASE_DIR, DATA_FILE)
if not os.path.isabs(INDEX_FILE):
    INDEX_FILE = os.path.join(BASE_DIR, INDEX_FILE)

# Global variables
original_data = None
p_loaded = None
process = None
reranker = None
_initialized = False


def initialize_once(api_key=None):
    global _initialized, original_data, p_loaded, process, reranker

    if _initialized:
        return
    
    load_dotenv()

    if api_key:
        print("API key provided but not needed for OpenRouter (uses OPENROUTER_API_KEY from .env)")

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
        p_loaded.load_index(INDEX_FILE, max_elements=len(original_data))
        p_loaded.set_ef(100)
        p_loaded.set_num_threads(4)  # Thêm multi-threading
        print(f"✓ Tải index HNSW thành công (Dimensions: {num_dimensions}).")
    except Exception as e:
        print(f"[LỖI] Không thể tải index: {e}")
        return

    print("Đang khởi tạo EmbeddingProcessor...")
    try:
        process = EmbeddingProcessor()
        print("✓ Khởi tạo EmbeddingProcessor thành công.")
    except Exception as e:
        print(f"[LỖI] Không thể khởi tạo EmbeddingProcessor: {e}")
        return

    print("Đang khởi tạo Reranker...")
    try:
        reranker = CrossEncoder('BAAI/bge-reranker-base')
        print("✓ Khởi tạo Reranker thành công.\n")
    except Exception as e:
        print(f"[LỖI] Không thể khởi tạo Reranker: {e}")
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


def search_index(query, k=20, timeout=10):
    if not p_loaded or not process or not original_data or not reranker:
        print("[LỖI] Hệ thống RAG chưa sẵn sàng.")
        return [], 0.0

    start_time = time.time()
    
    try:
        # Timeout cho embedding
        query_vector = process.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        
        if time.time() - start_time > timeout:
            print(f"[TIMEOUT] Embedding timeout")
            return [], 0.0

        # Timeout cho search
        labels, distances = p_loaded.knn_query(query_vector, k=k)
        similarities = 1 - distances[0]

        if time.time() - start_time > timeout:
            print(f"[TIMEOUT] Search timeout")
            return [], 0.0

        print(f"\n--- {k} kết quả tìm kiếm ban đầu ---")

        docs = []
        for i, label_id in enumerate(labels[0]):
            item = original_data[label_id].copy()
            item['similarity_score'] = float(similarities[i])
            docs.append(item)

        # RERANKING với timeout
        print("Đang rerank...")
        pairs = [[query, doc['content'][:512]] for doc in docs]  # Giới hạn content length
        rerank_scores = reranker.predict(pairs)
        
        if time.time() - start_time > timeout:
            print(f"[TIMEOUT] Rerank timeout, trả về kết quả ban đầu")
            return docs[:10], docs[0]['similarity_score'] if docs else 0.0
        
        for doc, score in zip(docs, rerank_scores):
            doc['rerank_score'] = float(score)
        
        docs.sort(key=lambda x: x['rerank_score'], reverse=True)
        top_docs = docs[:10]
        
        print(f"\n--- Top 10 sau reranking ({time.time() - start_time:.2f}s) ---")
        if top_docs:
            print(f"Rerank score cao nhất: {top_docs[0]['rerank_score']:.3f}")
            
            for doc in top_docs:
                print(f"[Rerank: {doc['rerank_score']:.3f} | Sim: {doc['similarity_score']:.3f}]")
                print(f"  ID: {doc.get('id', 'N/A')}, Title: {doc.get('title', 'N/A')}")
                print(f"  Content: {doc.get('content', '')[:50]}...")

            return top_docs, top_docs[0]['rerank_score']
        else:
            return [], 0.0
        
    except Exception as e:
        print(f"[LỖI] Lỗi khi tìm kiếm ({time.time() - start_time:.2f}s): {e}")
        return [], 0.0