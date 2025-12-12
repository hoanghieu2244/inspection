# AI Evidence Finder & Report Generator (Prototype)

Mục tiêu: hệ thống nhỏ giúp:
- Index các văn bản/chứng cứ (doc_id, text).
- Với 1 yêu cầu/tiêu chí, hệ thống tự tìm những minh chứng liên quan (RAG) và sinh báo cáo tiếng Việt kèm tham chiếu minh chứng.

Yêu cầu trước khi chạy:
- Python 3.11+
- Biến môi trường: OPENAI_API_KEY

Cài đặt & chạy:
1. Tạo virtualenv & cài:
   pip install -r requirements.txt

2. Tạo file .env với:
   OPENAI_API_KEY=sk-...

3. Chạy server:
   uvicorn app.main:app --reload --port 8000

Endpoints:
- POST /index
  body: { "documents": [ { "id":"doc1", "text":"..." }, ... ] }
  Mục đích: index các chứng cứ vào FAISS.

- POST /report
  body: { "query": "Yêu cầu / tiêu chí cần đánh giá", "top_k": 5 }
  Trả về: {
    "report": "...",               # báo cáo tiếng Việt
    "evidences": [ {id, score, snippet}, ... ],
    "_audit": { model, raw_output }
  }

Lưu ý:
- Đây là prototype in-memory FAISS với persist ra disk. Để production, chuyển sang Pinecone/Milvus + persist metadata.
- Thêm OCR nếu nguồn dữ liệu là PDF/ảnh; map page-level fragments để tham chiếu chính xác.
- Mask PII trước khi gửi lên API bên thứ ba.
