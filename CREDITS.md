# Credits và license

Mọi model, thư viện và dữ liệu mà AI service dùng, ai làm ra và được dùng theo điều kiện nào.
License của model lấy từ model card trên Hugging Face (kiểm tra ngày 28/10/2026). Trước khi dùng cho mục đích
khác, hãy đọc lại bản gốc qua link.

## Model đang chạy trong service

| Model | Dùng để | Tác giả | License |
| --- | --- | --- | --- |
| [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | embedding cho tìm kiếm ngữ nghĩa | UKP Lab, TU Darmstadt (Reimers & Gurevych) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [5CD-AI/Vietnamese-Sentiment-visobert](https://huggingface.co/5CD-AI/Vietnamese-Sentiment-visobert) | cảm xúc review (POS / NEU / NEG) | 5CD-AI | **không công bố**; model nền chỉ cho nghiên cứu (xem dưới) |
| `best.pt`, fine-tune từ [Ultralytics YOLO11m](https://docs.ultralytics.com/models/yolo11/) | nhận diện 5 món từ ảnh | Ultralytics; nhóm VietNomNom fine-tune | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) |
| [Gemini API](https://ai.google.dev/gemini-api) (`llm_parser.py`, tuỳ chọn) | đọc các câu hỏi mà luật không hiểu | Google | [Gemini API Terms](https://ai.google.dev/gemini-api/terms); không phải mã nguồn mở, chỉ bật khi đặt `GEMINI_API_KEY`, sẽ được thay bằng parser cục bộ |

**Về model sentiment.** `5CD-AI/Vietnamese-Sentiment-visobert` được fine-tune từ
[5CD-AI/visobert-14gb-corpus](https://huggingface.co/5CD-AI/visobert-14gb-corpus), vốn dựa trên
[uitnlp/visobert](https://huggingface.co/uitnlp/visobert). Tác giả ViSoBERT ghi rõ *"Our ViSoBERT model is available
only for research purposes"* và yêu cầu trích dẫn bài báo khi dùng:

> Quoc-Nam Nguyen, Thang Chau Phan, Duc-Vu Nguyen, Kiet Van Nguyen. 2023.
> [ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing](https://aclanthology.org/2023.emnlp-main.315/).
> EMNLP 2023.

Dữ liệu fine-tune của 5CD-AI gồm SA-VLSP2016, AIVIVN-2019, UIT-VSFC, UIT-VSMEC, UIT-ViCTSD, UIT-ViOCD, UIT-ViSFD,
Vi-amazon-reviews và Tiki-reviews (theo model card).

**Về YOLO.** Weights fine-tune từ YOLO11 là sản phẩm phái sinh nên cũng theo AGPL-3.0: nếu cung cấp service qua
mạng thì phải công khai mã nguồn cho người dùng. Repo này công khai nên đáp ứng điều kiện đó. Muốn dùng thương mại
mà không mở mã nguồn thì cần [Ultralytics Enterprise License](https://www.ultralytics.com/license).

> Glenn Jocher, Jing Qiu. 2024. *Ultralytics YOLO11*. https://github.com/ultralytics/ultralytics

## Model chạy offline trên Kaggle

| Model | Dùng để | Tác giả | License |
| --- | --- | --- | --- |
| [timm/ViT-gopt-16-SigLIP2-384](https://huggingface.co/timm/ViT-gopt-16-SigLIP2-384) (`kaggle_photo_index.py`) | phân loại ảnh bìa quán (món ăn / logo / menu…) | Google | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |

> Michael Tschannen et al. 2025. [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic
> Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786).

## Model dự kiến dùng khi fine-tune ([kaggle/README.md](kaggle/README.md))

| Model | Vai trò | License |
| --- | --- | --- |
| [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) (bản [QAT q4_0 GGUF](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf)) | sinh dữ liệu train, gán nhãn silver | [Apache-2.0](https://ai.google.dev/gemma/docs/gemma_4_license) |
| [AITeamVN/Vietnamese_Embedding_v2](https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2) | ứng viên embedding (bge-m3 fine-tune tiếng Việt) | Apache-2.0 |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | ứng viên embedding | MIT |
| [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | ứng viên embedding | Apache-2.0 |
| [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) | ứng viên embedding | Apache-2.0 |
| [contextboxai/halong_embedding](https://huggingface.co/contextboxai/halong_embedding) | ứng viên embedding (e5 fine-tune tiếng Việt) | Apache-2.0 |
| [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | ứng viên reranker | Apache-2.0 |
| [Alibaba-NLP/gte-multilingual-reranker-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base) | ứng viên reranker | Apache-2.0 |
| [Horizon-Labs/multilingual-reranker-small](https://huggingface.co/Horizon-Labs/multilingual-reranker-small) | ứng viên reranker | Apache-2.0 |
| [namdp-ptit/ViRanker](https://huggingface.co/namdp-ptit/ViRanker) | ứng viên reranker tiếng Việt | Apache-2.0 |
| [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) | nền thay ViSoBERT nếu cần license mở | MIT |
| [VLSP 2018 ABSA](https://vlsp.org.vn/resources) (dữ liệu) | train ABSA mảng nhà hàng | theo thoả thuận sử dụng của VLSP, chỉ cho nghiên cứu |

## Thư viện

| Thư viện | License |
| --- | --- |
| [FastAPI](https://github.com/fastapi/fastapi), [Pydantic](https://github.com/pydantic/pydantic) | MIT |
| [Uvicorn](https://github.com/encode/uvicorn) | BSD-3-Clause |
| [PyTorch](https://github.com/pytorch/pytorch) | BSD-3-Clause |
| [Transformers](https://github.com/huggingface/transformers), [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers) | Apache-2.0 |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn), [pandas](https://github.com/pandas-dev/pandas), [NumPy](https://github.com/numpy/numpy) | BSD-3-Clause |
| [PyMongo](https://github.com/mongodb/mongo-python-driver), [OpenCV](https://github.com/opencv/opencv) | Apache-2.0 |
| [Pillow](https://github.com/python-pillow/Pillow) | MIT-CMU |
| [open_clip](https://github.com/mlfoundations/open_clip) (Kaggle) | MIT |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) (Kaggle) | MIT |

## Dữ liệu

- **Quán và review** được crawl từ [Foody.vn](https://www.foody.vn) / ShopeeFood. Bản quyền nội dung thuộc về Foody
  và người viết review; ảnh quán được hiển thị trực tiếp từ CDN của ShopeeFood. Dữ liệu chỉ dùng cho mục đích học
  tập: không public dataset review, không đẩy model train trên review lên Hub ở chế độ public.
- **Ảnh train YOLO** (`data.yaml`, 5 lớp: Bánh mì, Bột chiên, Bún, Gỏi cuốn, Phở): *nhóm bổ sung nguồn dataset và
  license tại đây.*

## Muốn dùng thương mại

Có ba thứ cần thay hoặc xin phép: model sentiment (ViSoBERT chỉ cho nghiên cứu, thay bằng model nền license mở như
xlm-roberta-base rồi fine-tune lại), YOLO (AGPL-3.0: mở mã nguồn hoặc mua Enterprise License), và dữ liệu Foody
(cần thoả thuận với Foody / ShopeeFood).
