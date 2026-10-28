---
title: VietNomNom AI Service
emoji: 🍜
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# VietNomNom — AI Service

Service FastAPI lo toàn bộ phần AI của VietNomNom: tìm quán bằng ngôn ngữ tự nhiên, chatbot nhiều lượt, hỏi đáp về
từng quán dựa trên review, phân tích cảm xúc review và nhận diện món ăn từ ảnh. Mọi model đều là mã nguồn mở và chạy
ngay trong service.

- Chạy thật: <https://sirin2412-vietnomnom-ai.hf.space> (HF Space, CPU basic); mở `/docs` để thử API
- Backend: [Project-Smart-Food-Recommendation-Backend](https://github.com/ddanh2436/Project-Smart-Food-Recommendation-Backend)
- Frontend: [Project-Smart-Food-Recommendation-Frontend](https://github.com/ddanh2436/Project-Smart-Food-Recommendation-Frontend)

```text
Frontend (Vercel) → Backend (Render) → AI service (HF Space) → MongoDB Atlas
```

Chỉ Backend gọi service này. Khi đặt `INTERNAL_API_TOKEN`, mọi request phải có header `x-internal-token` trùng khớp.

## Tìm kiếm hoạt động thế nào

Lọc trước, xếp hạng sau:

1. **Hiểu câu hỏi** (`intent.py`, `knowledge.py`): tách câu thành món, quận/thành phố, khoảng giá ("150k/người"),
   điểm tối thiểu, buổi ("tối nay"), số người ("nhóm 5 người"), món cần tránh và kiểu sắp xếp. Câu mà luật không
   đọc được mới chuyển cho LLM parser (`llm_parser.py`, chỉ bật khi có `GEMINI_API_KEY`).
2. **Lọc cứng** (`search.py`): quận, giá, điểm, giờ mở cửa theo giờ Việt Nam (UTC+7), bỏ quán suất 1 người khi đi
   nhóm. Bộ lọc nào làm kết quả rỗng thì được nới ra, và tên bộ lọc đó nằm trong `relaxed_filters`.
3. **Bán kính quanh người dùng**: khi có GPS, ưu tiên quán trong 15 km, không đủ thì mở rộng lên 50 km. Hai mốc
   này không áp dụng khi người dùng đã nêu rõ khu vực.
4. **Xếp hạng lai**: khớp tên, khớp tag món, TF-IDF n-gram ký tự (gõ không dấu "bun bo" vẫn ra "bún bò"), độ
   tương đồng embedding, cộng thêm điểm đã hiệu chỉnh và khoảng cách.

Điểm hiển thị ở mọi nơi là **điểm đã hiệu chỉnh** (`diemTrungBinhAdj`, Backend tính): điểm gốc được kéo về mức
trung bình toàn hệ thống tuỳ theo số review, để quán có vài review 10 điểm không vượt quán có lịch sử dài.

## Model

| Thành phần | Model | Ghi chú |
| --- | --- | --- |
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` | đổi qua `EMBEDDING_MODEL`; vector được cache trong `embeddings_cache.npz` |
| Cảm xúc review | `5CD-AI/Vietnamese-Sentiment-visobert` | 3 nhãn POS/NEU/NEG; bản 4 nhãn (thêm "trái chiều") xem notebook 7 |
| Nhận diện món | `best.pt` (YOLO11m tự train) | 5 lớp: Bánh mì, Bột chiên, Bún, Gỏi cuốn, Phở |
| Tìm kiếm từ vựng | TF-IDF n-gram ký tự (scikit-learn) | |

Kế hoạch fine-tune và encode lại các model trên dữ liệu của web (Kaggle GPU miễn phí) nằm trong
[kaggle/README.md](kaggle/README.md).

## Endpoint

| Method | Path | Mục đích |
| --- | --- | --- |
| `GET` | `/health` | trạng thái từng model và dữ liệu |
| `POST` | `/recommend` | tìm quán (lọc rồi xếp hạng) |
| `POST` | `/parse-intent` | xem câu hỏi được hiểu thành những gì |
| `POST` | `/chat` | chatbot nhiều lượt, nhận `history` |
| `POST` | `/review-insights` | tóm tắt review theo khía cạnh của một quán |
| `POST` | `/sentiment`, `/sentiment/batch` | cảm xúc của một review / tối đa 256 review |
| `POST` | `/predict-food`, `/search-by-image` | nhận diện món từ ảnh / nhận diện rồi tìm quán |
| `GET` `POST` | `/admin/aspect-index` | xem / chạy tính khía cạnh review (cần `x-admin-token`) |
| `POST` | `/admin/reload` | tải lại dữ liệu và index (cần `x-admin-token`) |

## Cấu hình

Chạy ở máy thì chép `.env.example` thành `.env`. Trên HF Space thì đặt ở **Settings → Variables and secrets**
(các giá trị bí mật để dạng *Secret*).

| Biến | Bắt buộc | Ghi chú |
| --- | --- | --- |
| `MONGO_URI` | có | chuỗi kết nối MongoDB Atlas |
| `INTERNAL_API_TOKEN` | nên có | trùng với Backend; chặn mọi request không qua Backend |
| `ADMIN_TOKEN` | | cho các route `/admin/*` |
| `DB_NAME`, `COLLECTION_NAME`, `REVIEWS_COLLECTION` | | mặc định `VietNomNom` và tên collection hiện tại |
| `PORT`, `HOST`, `CORS_ORIGINS` | | Space dùng cổng `7860` |
| `ENABLE_SEMANTIC_SEARCH`, `ENABLE_SENTIMENT`, `ENABLE_YOLO` | | `false` để tắt từng model khi thiếu RAM |
| `EMBEDDING_MODEL`, `EMBEDDING_QUERY_PREFIX`, `EMBEDDING_PASSAGE_PREFIX`, `EMBEDDING_TRUST_REMOTE_CODE` | | đổi model embedding; e5 cần prefix "query: " / "passage: " |
| `GEMINI_API_KEY`, `LLM_MODEL`, `LLM_DAILY_CAP`, `LLM_CONFIDENCE_GATE`, `LLM_TIMEOUT_SECONDS` | | LLM parser dự phòng; không đặt key thì tắt hẳn |
| `DATA_REFRESH_SECONDS`, `MAX_RESULTS`, `MAX_UPLOAD_MB`, `YOLO_MODEL_PATH` | | |

## Chạy ở máy

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
cp .env.example .env                                    # điền MONGO_URI
python api.py                                           # http://127.0.0.1:5000/docs
```

Lần chạy đầu sẽ tải model sentiment và embedding (khoảng 600 MB) nên khá chậm. Những lần sau đọc từ cache.

## Kiểm thử và đo chất lượng

```bash
python test_ranking.py                 # self-check: bán kính, cache embedding, nhãn khen/chê, giờ mở cửa, bộ lọc
python benchmark.py                    # đo tìm kiếm trên search_eval.json: nDCG@10, recall@10, MRR@10
python build_qrels.py                  # dựng lại search_qrels.json (đáp án theo ID quán)
```

`qrels_to_grade.json` là file chấm tay: chỉ gồm những câu mà luật không tự chấm được, mỗi quán một dòng kèm quận,
giá, điểm, giờ mở cửa và tỉ lệ review khen. Đổi `"grade": null` thành 0–3, rồi chạy `build_qrels.py` để gộp điểm vào
`search_qrels.json`.

## Tính khía cạnh review

`aspect_index.py` tính tỉ lệ khen/chê cho 6 khía cạnh (món ăn, giá, phục vụ, không gian, vệ sinh, chỗ để xe) rồi ghi
vào từng quán. Trang quán và bộ xếp hạng đọc thẳng các giá trị này.

```bash
python aspect_index.py --status
python aspect_index.py --all                                         # chỉ quán có review mới
python aspect_index.py --before 2026-10-27T14:33:00+00:00 --all      # tính lại mọi quán tính trước mốc này
```

Lượt chạy `--before` dừng giữa chừng thì chạy lại với cùng mốc thời gian là chạy tiếp được. Tính toàn bộ khoảng
5.300 quán mất vài giờ trên CPU.

## Triển khai lên HF Space

Lịch sử git của Space khác với GitHub, nên cập nhật bằng cách upload các file đã đổi (tab **Files**), rồi
**Restart Space**. Service chỉ đọc dữ liệu quán lúc khởi động, nên sau khi chạy aspect index cũng cần restart.
Xem thêm [DEPLOY.md](DEPLOY.md).

## Cấu trúc

| File | Vai trò |
| --- | --- |
| `api.py` | ứng dụng FastAPI, route, schema |
| `config.py` | cấu hình từ biến môi trường |
| `knowledge.py`, `text_utils.py` | từ điển món/địa danh, chuẩn hoá chữ, khoảng cách, giá |
| `intent.py`, `llm_parser.py` | câu hỏi → các trường có cấu trúc |
| `data_store.py` | snapshot MongoDB, index TF-IDF và embedding |
| `search.py` | lọc rồi xếp hạng |
| `chat.py` | chatbot nhiều lượt |
| `place_qa.py` | trả lời câu hỏi về một quán cụ thể |
| `sentiment.py`, `review_insights.py`, `aspect_index.py` | cảm xúc review, tóm tắt theo khía cạnh |
| `vision.py`, `dish_labels.py`, `kaggle_photo_index.py` | nhận diện món, kiểm tra ảnh bìa quán (chạy offline trên Kaggle) |
| `benchmark.py`, `build_qrels.py`, `search_eval.json`, `search_qrels.json` | bộ đo chất lượng tìm kiếm |
| `kaggle/README.md` | hướng dẫn notebook fine-tune trên Kaggle |
