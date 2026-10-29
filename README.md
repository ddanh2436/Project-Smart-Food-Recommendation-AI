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

## Kỹ thuật sử dụng

### Tìm kiếm: lọc trước, xếp hạng sau (`intent.py`, `search.py`)

1. **Hiểu câu hỏi bằng luật + từ điển** (`intent.py`, `knowledge.py`): tách câu thành món, quận/thành phố, khoảng
   giá ("150k/người"), điểm tối thiểu, buổi ("tối nay"), số người ("nhóm 5 người"), món cần tránh và kiểu sắp xếp.
   Các khái niệm như "món nước" hay "trời lạnh" được mở rộng thành danh sách món. Chỉ câu mà luật không đọc được mới
   chuyển cho LLM parser (`llm_parser.py`), và LLM chỉ được điền các trường có sẵn, không được tự nêu tên quán.
2. **Lọc cứng có nới dần**: quận, giá, điểm, giờ mở cửa theo giờ Việt Nam (UTC+7), bỏ quán suất 1 người khi đi
   nhóm. Bộ lọc nào làm kết quả rỗng thì được nới ra và báo lại trong `relaxed_filters`.
3. **Bán kính quanh người dùng**: khoảng cách haversine, ưu tiên quán trong 15 km, không đủ 5 quán thì mở lên
   50 km. Không áp dụng khi người dùng đã nêu khu vực.
4. **Xếp hạng lai**:
   - khớp tên và tag món theo ranh giới từ;
   - **TF-IDF n-gram ký tự** trên chữ đã bỏ dấu, nên gõ "bun bo" vẫn ra "bún bò";
   - **embedding đa ngôn ngữ**, tính một lần cho mọi quán rồi cache theo model và nội dung (`embeddings_cache.npz`);
   - cộng thêm điểm đã hiệu chỉnh và khoảng cách, rồi **đa dạng hoá** để top kết quả không toàn một món.
5. **Giải thích kết quả**: mỗi quán kèm lý do (khớp món, đúng giá, điểm, khoảng cách) và lưu ý (sắp đóng cửa…).

Điểm hiển thị ở mọi nơi là **điểm đã hiệu chỉnh** (`diemTrungBinhAdj`, Backend tính bằng trung bình Bayes): điểm
gốc được kéo về mức trung bình toàn hệ thống tuỳ theo số review, để quán có vài review 10 điểm không vượt quán có
lịch sử dài.

### Chatbot (`chat.py`)

- **Nhiều lượt**: gộp ý định của câu mới với câu trước, nên "rẻ hơn đi" hay "còn quán nào khác" sẽ sửa lại lần tìm
  trước thay vì tìm lại từ đầu.
- **Hỏi lại khi câu quá mơ hồ** ("đói quá"): trả về chip gợi ý món, giá, khu vực để người dùng bấm chọn.
- **Câu trả lời chỉ dùng dữ liệu thật**: nói đúng số quán tìm được, giải thích vì sao ít kết quả, nhắc gọi đặt bàn
  khi đi nhóm, và nói "chưa có dữ liệu" thay vì bịa.
- **Hết quán trong quận** thì tìm tiếp quanh những quán đã gợi ý, không lặp lại quán cũ.

### Hiểu review (`review_insights.py`, `aspect_index.py`, `place_qa.py`)

- **ABSA không cần LLM**:
  - tách review thành mệnh đề, vì người Việt hay khen và chê trong cùng một câu ("ngon nhưng phục vụ chậm");
  - gán mệnh đề vào 6 khía cạnh (món ăn, giá, phục vụ, không gian, vệ sinh, chỗ để xe) bằng từ khoá;
  - chấm cảm xúc từng mệnh đề bằng ViSoBERT, chạy theo batch.
- **Một quy tắc nhãn duy nhất**: khen ≥ 70% là "được khen", khen ≤ 30% và chê nhiều hơn khen là "bị phàn nàn",
  còn lại là "ý kiến trái chiều". Khía cạnh có dưới 3 lượt nhắc thì không kết luận.
- **Tính trước cho cả hệ thống** (`aspect_index.py`): ghi kết quả vào từng quán, chạy tăng dần và tiếp tục được
  nếu bị ngắt. Nhờ vậy câu như "quán sạch sẽ" được trả lời bằng bằng chứng, không phải khớp chữ.
- **Hỏi đáp về một quán** (`place_qa.py`): trả lời "giá ở đây thế nào?" bằng số liệu và câu trích nguyên văn từ
  review, kèm điểm chung của quán khi khác điểm con.

### Ảnh (`vision.py`, `kaggle_photo_index.py`)

- **YOLO11m fine-tune** nhận diện 5 món, chia kết quả theo độ chắc chắn (chắc chắn / có thể là / chỉ đoán được
  nhóm món) để luôn có gợi ý bấm tiếp.
- **SigLIP2 zero-shot** (chạy offline trên Kaggle) phân loại ảnh bìa của 5.506 quán thành món ăn, logo, menu, biển
  hiệu… Ảnh không phải món ăn được thay bằng ảnh trung tính.

### Đo chất lượng (`benchmark.py`, `build_qrels.py`)

Bộ đo gồm 136 câu tìm kiếm với đáp án theo **ID quán**, chấm theo thang 0–3. Phần lớn đáp án do luật chấm trên toàn
bộ quán, phần còn lại lấy top 10 rồi chấm tay (pooling). Chỉ số là nDCG@10, recall@10 và MRR@10, cộng thêm các nhóm
kiểm tra riêng (khoảng cách, câu khó, câu phải trả rỗng). Model mới chỉ được thay vào khi tốt hơn trên bộ đo này.

## Model

| Thành phần | Model | License |
| --- | --- | --- |
| Embedding | [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | Apache-2.0 |
| Cảm xúc review | [5CD-AI/Vietnamese-Sentiment-visobert](https://huggingface.co/5CD-AI/Vietnamese-Sentiment-visobert) (3 nhãn), fine-tune thành `Sirin2412/vietnomnom-sentiment-4c` (thêm MIX "trái chiều", private) | không công bố; nền ViSoBERT chỉ cho nghiên cứu |
| Nhận diện món | `best.pt`, fine-tune từ [YOLO11m](https://docs.ultralytics.com/models/yolo11/) | AGPL-3.0 |
| Phân loại ảnh bìa (offline) | [SigLIP2 ViT-gopt-16-384](https://huggingface.co/timm/ViT-gopt-16-SigLIP2-384) | Apache-2.0 |
| Parser dự phòng (tuỳ chọn) | Gemini API | điều khoản của Google; sẽ thay bằng parser cục bộ |
| Tìm kiếm từ vựng | TF-IDF n-gram ký tự (scikit-learn) | BSD-3-Clause |

**Credits, trích dẫn và license đầy đủ của từng model, thư viện và dữ liệu: [CREDITS.md](CREDITS.md).**

Kế hoạch fine-tune và encode lại các model trên dữ liệu của web (Kaggle GPU miễn phí, gán nhãn bằng Gemma 4 E4B,
sentiment 4 nhãn có thêm "trái chiều") nằm trong [kaggle/README.md](kaggle/README.md).

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
| `relabel_sentiment.py` | gán lại nhãn cảm xúc cho mọi review khi đổi `SENTIMENT_MODEL` |
| `vision.py`, `dish_labels.py`, `kaggle_photo_index.py` | nhận diện món, kiểm tra ảnh bìa quán (chạy offline trên Kaggle) |
| `benchmark.py`, `build_qrels.py`, `search_eval.json`, `search_qrels.json` | bộ đo chất lượng tìm kiếm |
| `kaggle/README.md` | hướng dẫn notebook fine-tune trên Kaggle |
