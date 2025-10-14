import json
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any


class EmbeddingProcessor:
    """
    Class để thực hiện embedding cho các tài liệu y khoa đã được xử lý.
    Sử dụng mô hình chuyên biệt cho tiếng Việt để đạt hiệu quả cao nhất.
    """

    def __init__(self, model_name: str = 'bkai-foundation-models/vietnamese-bi-encoder'):
        """
        Khởi tạo và tải mô hình embedding.
        Tự động chọn GPU (cuda) nếu có, nếu không thì dùng CPU.
        """
        # Xác định thiết bị (GPU hoặc CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Sử dụng thiết bị: {self.device}")

        # Tải mô hình từ Hugging Face và chuyển nó lên thiết bị đã chọn
        print(f"Đang tải mô hình embedding '{model_name}'...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print("✓ Tải mô hình thành công!")

    def load_processed_data(self, input_file: str = "data_clean/medical_data.json") -> List[Dict[str, Any]]:
        """Tải dữ liệu đã xử lý từ file JSON."""
        print(f"Đang đọc dữ liệu từ '{input_file}'...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Đọc thành công {len(data)} tài liệu.")
        return data

    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Tạo vector embedding cho danh sách các tài liệu.
        Kết hợp tiêu đề và nội dung để có ngữ cảnh tốt hơn.
        """
        # 1. Chuẩn bị các đoạn text cần embedding
        #    Kết hợp tiêu đề và nội dung giúp vector nắm bắt được thông tin tổng quát nhất.
        texts_to_embed = [
            f"Tiêu đề: {doc.get('title', '')}\nNội dung: {doc.get('content', '')}"
            for doc in documents
        ]

        print(f"Bắt đầu quá trình embedding cho {len(texts_to_embed)} tài liệu. Quá trình này có thể mất vài phút...")

        # 2. Thực hiện embedding
        #    Hàm model.encode() sẽ xử lý theo batch để tối ưu tốc độ.
        embeddings = self.model.encode(
            texts_to_embed,
            show_progress_bar=True,
            convert_to_numpy=True  # Trả về dưới dạng numpy array để xử lý nhanh hơn
        )

        print("✓ Quá trình embedding hoàn tất!")

        # 3. Thêm vector embedding vào mỗi tài liệu
        for i, doc in enumerate(documents):
            # Chuyển numpy array thành list để có thể lưu ra file JSON
            doc['embedding'] = embeddings[i].tolist()

        return documents

    def save_embedded_data(self, documents: List[Dict[str, Any]],
                           output_file: str = "medical_data_with_embeddings.json"):
        """Lưu dữ liệu đã được embedding ra file JSON mới."""
        print(f"Đang lưu dữ liệu đã embedding vào '{output_file}'...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        print(f"✓ Đã lưu thành công {len(documents)} tài liệu.")


def main():
    """Hàm chính để chạy toàn bộ quy trình"""
    processor = EmbeddingProcessor()

    # Tải dữ liệu đã qua xử lý
    documents = processor.load_processed_data("data_clean/medical_data.json")

    if not documents:
        print("❌ Không tìm thấy tài liệu nào để xử lý.")
        return

    # Thực hiện embedding
    documents_with_embeddings = processor.embed_documents(documents)

    # Lưu kết quả
    processor.save_embedded_data(documents_with_embeddings)

    # Hiển thị ví dụ kết quả
    if documents_with_embeddings:
        print("\n" + "=" * 60)
        print("VÍ DỤ KẾT QUẢ SAU KHI EMBEDDING:")
        print("=" * 60)
        first_doc = documents_with_embeddings[0]
        print(f"📄 Tiêu đề: {first_doc['title']}")
        embedding_preview = first_doc['embedding'][:5]  # Chỉ hiển thị 5 số đầu tiên của vector
        print(f"📊 Vector embedding (5 chiều đầu tiên): {embedding_preview}...")
        print(f"📏 Tổng số chiều của vector: {len(first_doc['embedding'])}")
        print("=" * 60)


if __name__ == "__main__":
    main()