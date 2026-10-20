# Visual RAG — Chatbot Y tế (MSD) + 3D Geometry Viewer

Chatbot RAG tiếng Việt trên dữ liệu y khoa MSD: truy hồi bằng hnswlib + reranker CrossEncoder, trả lời qua LLM (OpenRouter). Kèm module **viewer hình học 3D tương tác** (three.js) theo mô hình spec → kernel → kiểm tra → render.

## Yêu cầu

- Python 3.12
- **Git LFS** — file dữ liệu lõi (`data_clean/medical_data_with_embeddings.json`, ~824MB) và index hnsw nằm trong LFS. Clone xong bắt buộc chạy:

```bash
git lfs install
git lfs pull
```

Thiếu bước này, các file dữ liệu chỉ là con trỏ 134 byte và app sẽ báo lỗi không tải được dữ liệu.

## Cài đặt & chạy

```bash
pip install -r requirements.txt
cp .env.example .env   # điền OPENROUTER_API_KEY và các biến cần thiết
python build_index.py  # (khi cần dựng lại index)
uvicorn api:app --host 0.0.0.0 --port 8000   # hoặc: python api.py
python main.py         # bản CLI
```

## Cấu trúc

- `api.py` — FastAPI app (chat streaming, quản lý session, endpoint geometry3d)
- `main.py` — chatbot dòng lệnh
- `src/` — lõi RAG: `config.py` (khởi tạo index/dữ liệu), `generative_model.py` (LLM), `query_rewriter.py`, `data_processor.py`, `conversation_manager.py`
- `src/geometry3d/` — kernel hình học + validator + preset cho viewer 3D
- `static/geo3d/` — trang viewer three.js, mở tại `http://localhost:8000/geo3d/`
- `src/finetune/` — sub-project finetune Bloomz (Unsloth), độc lập với chatbot
- `data_msd/` — dữ liệu MSD thô (txt); `data_clean/` — dữ liệu đã xử lý + embeddings + index

## Lưu ý vận hành

- Thư mục sinh ra khi chạy/huấn luyện (`results/`, `chroma_conversations/`, `unsloth_compiled_cache/`) **không** thuộc về git — đã được ignore.
- Adapter model đã finetune nằm ở `medical_chatbot/` (bản trong `src/finetune/checkpoint/` sẽ được dọn ở giai đoạn tái cấu trúc tiếp theo).
