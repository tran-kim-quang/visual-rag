# embed_data.py
import json
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

class EmbeddingProcessor:
    def __init__(self, model_name: str = 'hiieu/halong_embedding'):
        """
        Khởi tạo và tải mô hình embedding.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Sử dụng thiết bị: {self.device}")

        print(f"Đang tải mô hình embedding '{model_name}'...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print("✓ Tải mô hình thành công!")

    def load_processed_data(self, input_file: str = "data_clean/medical_data.json") -> List[Dict[str, Any]]:
        """Tải dữ liệu đã xử lý từ file JSON."""
        print(f"Đang đọc dữ liệu từ '{input_file}'...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Đọc thành công {len(data)} tài liệu gốc.")
        return data

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chia nhỏ 'content' của mỗi tài liệu thành các đoạn (chunks).
        Mỗi chunk sẽ trở thành một tài liệu mới để embed.
        """
        print("Đang bắt đầu quá trình chunking (cắt nhỏ) tài liệu...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " "]
        )

        all_chunks = []
        for i, doc in enumerate(documents):
            if not doc.get('content'):
                print(f"[Warning] Bỏ qua tài liệu {doc.get('id')} vì không có content.")
                continue

            chunks = text_splitter.split_text(doc['content'])

            for j, chunk_content in enumerate(chunks):
                new_chunk_doc = {
                    "id": f"{doc.get('id', 'doc')}_{i}_chunk_{j}",
                    "title": doc.get('title', ''),
                    "url": doc.get('url', ''),
                    "content": chunk_content
                }
                all_chunks.append(new_chunk_doc)

        print(f"✓ Đã chia {len(documents)} tài liệu gốc thành {len(all_chunks)} chunks.")
        return all_chunks

    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Tạo vector embedding cho danh sách các tài liệu (chunks).
        """
        texts_to_embed = [
            f"Tiêu đề: {doc.get('title', '')}\nNội dung: {doc.get('content', '')}"
            for doc in documents
        ]

        print(f"Bắt đầu quá trình embedding cho {len(texts_to_embed)} chunks...")

        embeddings = self.model.encode(
            texts_to_embed,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        print("✓ Quá trình embedding hoàn tất!")

        for i, doc in enumerate(documents):
            doc['embedding'] = embeddings[i].tolist()

        return documents

    def save_embedded_data(self, documents: List[Dict[str, Any]],
                           output_file: str = "data_clean/medical_data_with_embeddings.json"):
        """Lưu dữ liệu chunk đã được embedding ra file JSON mới."""
        print(f"Đang lưu dữ liệu đã embedding vào '{output_file}'...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        print(f"✓ Đã lưu thành công {len(documents)} chunks.")


def main():
    """Hàm chính để chạy toàn bộ quy trình"""
    processor = EmbeddingProcessor()

    documents = processor.load_processed_data("data_clean/medical_data.json")

    if not documents:
        print("❌ Không tìm thấy tài liệu nào để xử lý.")
        return

    all_chunks = processor.chunk_documents(documents)

    if not all_chunks:
        print("❌ Không tạo được chunk nào từ tài liệu.")
        return

    chunks_with_embeddings = processor.embed_documents(all_chunks)

    processor.save_embedded_data(chunks_with_embeddings, "data_clean/medical_data_with_embeddings.json")

    if chunks_with_embeddings:
        print("\n" + "=" * 60)
        print("VÍ DỤ KẾT QUẢ CHUNK SAU KHI EMBEDDING:")
        print("=" * 60)
        first_chunk = chunks_with_embeddings[0]
        print(f"ID (của chunk): {first_chunk['id']}")
        print(f"Tiêu đề (gốc): {first_chunk['title']}")
        print(f"Nội dung (chunk): {first_chunk['content'][:150]}...")
        embedding_preview = first_chunk['embedding'][:5]
        print(f"Vector embedding (5 chiều đầu tiên): {embedding_preview}...")
        print(f"Tổng số chiều của vector: {len(first_chunk['embedding'])}")
        print("=" * 60)


if __name__ == "__main__":
    main()