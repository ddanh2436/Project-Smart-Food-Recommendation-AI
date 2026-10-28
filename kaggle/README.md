# Notebook Kaggle: sinh dữ liệu, fine-tune và encode (không dùng API trả phí)

Toàn bộ pipeline chạy trên **model mã nguồn mở**, train trên **Kaggle GPU miễn phí** và
chạy thật trên **HF Space CPU basic** (2 vCPU, 16 GB RAM). Không có bước nào gọi Gemini hay API
bên ngoài. Kể cả việc sinh dữ liệu train cũng do một LLM open-weight chạy ngay trên Kaggle đảm nhận.

| # | Notebook | GPU | Đầu ra |
| --- | --- | --- | --- |
| 1 | Chọn embedding và encode sẵn | T4 ×1 | `emb_<model>.npz` và bảng benchmark |
| 2 | Sinh dữ liệu bằng LLM open-weight (Gemma 4 E4B) | T4 ×1 | `pairs.jsonl`, `parser.jsonl`, `absa.jsonl` |
| 3 | Fine-tune embedding | T4 ×1 | model trên HF Hub và `.npz` mới |
| 4 | Fine-tune reranker | T4 ×1 | cross-encoder trên HF Hub |
| 5 | Parser cục bộ thay Gemini | T4 ×1 | classifier đa nhãn |
| 6 | ABSA (sentiment theo khía cạnh) | T4 ×1 | classifier 6 khía cạnh × 4 nhãn |
| 7 | Sentiment 4 nhãn cho từng review | T4 ×1 | classifier POS / NEG / MIX / NEU |

**Quy tắc chung:** mỗi model mới chỉ được giữ lại khi `benchmark.py --against baseline.json`
cho thấy nó tốt hơn (nDCG@10, MRR@10, nhóm `distance` và `hard`) và độ trễ trên CPU vẫn nằm trong ngân sách.

---

## 0. Cell chuẩn bị (dùng cho mọi notebook)

Trên Kaggle: *Add-ons → Secrets*, thêm `MONGO_URI`, và thêm `HF_TOKEN` (quyền write) nếu muốn đẩy
model lên Hub. Ở mục **Network access** của MongoDB Atlas, cần cho phép IP của Kaggle truy cập (Space đang chạy
được thì nhiều khả năng Atlas đã mở `0.0.0.0/0`).

```python
import os, subprocess, sys
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
os.environ["MONGO_URI"] = secrets.get_secret("MONGO_URI")
os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")      # bỏ nếu không đẩy lên Hub

!git clone -q https://github.com/ddanh2436/Project-Smart-Food-Recommendation-AI.git /kaggle/working/ai
# Dùng đúng phiên bản như requirements.txt. KHÔNG cài lại torch: Kaggle đã có sẵn bản GPU.
!pip install -q "transformers~=4.57.1" "sentence-transformers~=5.7.0" "pymongo~=4.15.0" \
    "fastapi~=0.121.0" "uvicorn[standard]~=0.38.0" "python-multipart~=0.0.20" requests
sys.path.insert(0, "/kaggle/working/ai")
os.chdir("/kaggle/working/ai")
```

Với `data_store`, đừng tự viết lại logic đọc dữ liệu mà hãy import thẳng từ repo. Lý do: vector chỉ được dùng lại
khi text **trùng từng ký tự** với text mà service tạo ra (`prefix + "tên. tags"`).

```python
os.environ.update(ENABLE_SEMANTIC_SEARCH="false", ENABLE_SENTIMENT="false", ENABLE_YOLO="false")
from data_store import store
store.reload()
frame = store.frame            # 5506 quán, đã chuẩn hoá: id, name, tags, district, city, price...
print(len(frame))
```

---

## 1. Chọn embedding và encode sẵn

Chạy toàn bộ service với từng model ứng viên rồi dùng `benchmark.py` để chấm. Cách này đo cả pipeline
(luật, TF-IDF, embedding, bán kính), không chỉ riêng embedding.

| Model | Kích thước | Prefix truy vấn | Ghi chú |
| --- | --- | --- | --- |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 118M | không | **baseline hiện tại** |
| `AITeamVN/Vietnamese_Embedding_v2` | 568M | không | bge-m3 fine-tune tiếng Việt, khoảng 1,1M triplet |
| `Qwen/Qwen3-Embedding-0.6B` | 600M | `Instruct: ...\nQuery:` | đa ngôn ngữ, MMTEB-R 64.6 |
| `Alibaba-NLP/gte-multilingual-base` | 305M | không | cần `EMBEDDING_TRUST_REMOTE_CODE=true` |
| `hiieu/halong_embedding` | ~278M | xem model card | model tiếng Việt nhỏ, ưu tiên tốc độ |

```python
import json, time, signal, requests

CANDIDATES = {
    "minilm":  {"EMBEDDING_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"},
    "aiteam2": {"EMBEDDING_MODEL": "AITeamVN/Vietnamese_Embedding_v2"},
    "qwen06":  {"EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B",
                "EMBEDDING_QUERY_PREFIX": "Instruct: Given a Vietnamese food search query, retrieve restaurants that match it\nQuery:"},
    "gte":     {"EMBEDDING_MODEL": "Alibaba-NLP/gte-multilingual-base", "EMBEDDING_TRUST_REMOTE_CODE": "true"},
}

def bench(slug, extra_env):
    env = {**os.environ, "ENABLE_SEMANTIC_SEARCH": "true", "ENABLE_SENTIMENT": "false",
           "ENABLE_YOLO": "false", "GEMINI_API_KEY": "", **extra_env,
           "EMBEDDING_CACHE_PATH": f"/kaggle/working/emb_{slug}.npz"}
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "api:app", "--port", "5000"], env=env)
    try:
        for _ in range(900):                     # lần đầu phải encode 5506 quán
            try:
                if requests.get("http://127.0.0.1:5000/health", timeout=2).json()["data"]["semantic_ready"]:
                    break
            except Exception:
                pass
            time.sleep(1)
        subprocess.run([sys.executable, "benchmark.py", "--save", f"/kaggle/working/bench_{slug}.json"],
                       env={**env, "BENCH_LLM_SPACING": "0", "AI_EVAL_URL": "http://127.0.0.1:5000"})
    finally:
        server.send_signal(signal.SIGINT); server.wait()

for slug, extra in CANDIDATES.items():
    bench(slug, extra)

for slug in CANDIDATES:
    m = json.load(open(f"/kaggle/working/bench_{slug}.json"))["metrics"]
    print(f"{slug:8} ndcg={m['ndcg@10']:6} mrr={m['mrr@10']:6} recall={m['recall@10']:6} top5={m['top5_accuracy']}")
```

**Đo độ trễ theo điều kiện CPU của Space**, vì GPU của Kaggle không phản ánh được môi trường thật:

```python
import torch
from sentence_transformers import SentenceTransformer
torch.set_num_threads(2)                         # Space free có 2 vCPU
for slug, extra in CANDIDATES.items():
    m = SentenceTransformer(extra["EMBEDDING_MODEL"], device="cpu",
                            trust_remote_code=extra.get("EMBEDDING_TRUST_REMOTE_CODE") == "true")
    q = extra.get("EMBEDDING_QUERY_PREFIX", "") + "bún bò quận 3 dưới 50k"
    m.encode([q]); t = time.time()
    for _ in range(20): m.encode([q])
    print(slug, f"{(time.time()-t)/20*1000:.0f} ms/query")
```

Cách chọn: lấy model có nDCG@10 cao nhất mà vẫn **dưới ~150 ms/query trên CPU**. Trên Space, file
`emb_<slug>.npz` của model đó sẽ được dùng ngay:

1. Copy file vào repo với tên khác `embeddings_cache.npz` (tên này nằm trong `.gitignore`), ví dụ `embeddings/aiteam2.npz`.
   File khoảng 25 MB nên khi push lên HF phải dùng LFS/Xet, giống cách `best.pt` đang làm.
2. Trên Space, đặt `EMBEDDING_MODEL`, các biến prefix (nếu có) và `EMBEDDING_CACHE_PATH=embeddings/aiteam2.npz`.
3. Khi khởi động, service chỉ encode những quán mới hoặc vừa bị sửa. Toàn bộ phần còn lại được load từ file.

---

## 2. Sinh dữ liệu bằng LLM open-weight (thay cho Gemini)

Dùng **Gemma 4 E4B** (Google DeepMind, Apache 2.0), bản lượng tử hoá QAT q4_0 do Google phát hành
(`google/gemma-4-E4B-it-qat-q4_0-gguf`, 5,15 GB). Model này xử lý tiếng Việt tốt hơn Qwen3/3.5 qua thử nghiệm thực tế.
Bản gốc BF16 nặng 16 GB, không vừa một T4 và T4 cũng không hỗ trợ bf16. Bản QAT 4-bit được huấn luyện để giữ chất
lượng gần với bản gốc, vừa một T4 và chạy bằng `llama.cpp`.

`llama-server` chạy nhiều request song song (`-np`) với continuous batching, nên vẫn đủ nhanh cho hàng chục nghìn
prompt. Output bị ép theo JSON schema giống như trước.

```python
# Build llama.cpp với CUDA cho T4 (compute capability 7.5): khoảng 10 phút, mỗi session một lần.
!git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /kaggle/working/llama.cpp
!cmake -S /kaggle/working/llama.cpp -B /kaggle/working/llama.cpp/build        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF > /dev/null
!cmake --build /kaggle/working/llama.cpp/build --target llama-server -j 4 > /dev/null

import json, subprocess, time, requests
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download

gguf = hf_hub_download("google/gemma-4-E4B-it-qat-q4_0-gguf", "gemma-4-E4B_q4_0-it.gguf")
SLOTS = 16                       # số request chạy song song; tràn VRAM thì giảm xuống 8
server = subprocess.Popen(
    ["/kaggle/working/llama.cpp/build/bin/llama-server", "-m", gguf, "-ngl", "99",
     "-c", str(2048 * SLOTS), "-np", str(SLOTS),       # mỗi slot 2048 token ngữ cảnh
     "--reasoning-budget", "0", "--port", "8080"],      # tắt chế độ suy nghĩ: chỉ cần nhãn
    stdout=subprocess.DEVNULL, stderr=open("/kaggle/working/llama.log", "w"))
for _ in range(120):                                     # chờ model nạp lên GPU
    try:
        if requests.get("http://127.0.0.1:8080/health", timeout=2).ok:
            break
    except requests.ConnectionError:
        pass
    time.sleep(5)

def _one(messages, schema, temperature, max_tokens):
    r = requests.post("http://127.0.0.1:8080/v1/chat/completions", timeout=600, json={
        "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
        "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}})
    return json.loads(r.json()["choices"][0]["message"]["content"])

def ask(prompts, schema, temperature=0.7, max_tokens=512):
    """Mỗi phần tử của prompts là [{"role": ..., "content": ...}]. Output bị ép theo JSON schema."""
    with ThreadPoolExecutor(SLOTS) as pool:
        return list(pool.map(lambda m: _one(m, schema, temperature, max_tokens), prompts))
```

> Server không lên thì xem `/kaggle/working/llama.log`. Hai lỗi hay gặp: tràn VRAM (giảm `SLOTS`) và bản llama.cpp
> quá cũ chưa hỗ trợ Gemma 4 (clone lại bản mới). `ask()` giữ nguyên giao diện, nên các bước 2a, 2b, 2c, 7b không đổi.

### 2a. Cặp (câu truy vấn, quán) cho embedding và reranker

Lấy vài review thật cho mỗi quán để câu truy vấn sinh ra phản ánh đúng những gì thực khách nói về quán:

```python
from pymongo import MongoClient
db = MongoClient(os.environ["MONGO_URI"])["VietNomNom"]
reviews = {}
for r in db["reviews"].find({}, {"urlGoc": 1, "noiDung": 1}):
    reviews.setdefault(r["urlGoc"], []).append(r["noiDung"][:300])
url_of = {str(d["_id"]): d.get("urlGoc") for d in db["restaurants"].find({}, {"urlGoc": 1})}

QUERY_SCHEMA = {"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"},
                "minItems": 5, "maxItems": 5}}, "required": ["queries"]}
SYSTEM = ("Bạn viết câu mà người Việt sẽ gõ vào ô tìm kiếm khi muốn tìm ĐÚNG loại quán này. "
          "Viết 5 câu khác kiểu nhau: (1) món + khu vực, (2) không dấu, viết tắt kiểu 'q1', '50k', "
          "(3) theo tâm trạng/thời tiết/dịp, (4) theo giá hoặc đối tượng (sinh viên, gia đình, hẹn hò), "
          "(5) mô tả đặc điểm khen trong review. KHÔNG được chứa tên quán hay số nhà.")

prompts = []
for row in frame.itertuples():
    snippet = "\n- ".join(reviews.get(url_of.get(row.id), [])[:3]) or "(chưa có review)"
    prompts.append([{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Quán: {row.name}\nTags: {row.tags}\nKhu vực: {row.district}, {row.city}\n"
                                                f"Giá: {row.price_text}\nReview:\n- {snippet}"}])
answers = ask(prompts, QUERY_SCHEMA)

import text_utils as tu
eval_queries = {tu.normalize(c["query"]) for c in json.load(open("search_eval.json", encoding="utf-8"))["cases"]}
with open("/kaggle/working/pairs.jsonl", "w", encoding="utf-8") as f:
    for row, a in zip(frame.itertuples(), answers):
        for q in a["queries"]:
            if tu.normalize(q) in eval_queries:      # chống rò rỉ: không train trên câu dùng để đo
                continue
            f.write(json.dumps({"query": q, "id": row.id}, ensure_ascii=False) + "\n")
```

Kết quả là khoảng 27 nghìn cặp. Trước khi train, nên đọc ngẫu nhiên khoảng 100 cặp: nếu một câu truy vấn đúng với *bất kỳ* quán phở nào
thì cũng không sao, vì bước 3 có lọc false negative.

### 2b. Nhãn cho parser cục bộ

Nên tái dùng **đúng prompt và từ vựng** mà Gemini đang dùng, để parser mới trả về đúng định dạng cũ:

```python
from llm_parser import PROMPT, DISHES, ADJECTIVES, ASPECTS, DISTRICTS, TIME_TAGS, SORTS
from intent import parse_intent

def enum_list(values): return {"type": "array", "items": {"type": "string", "enum": values}}
PARSER_SCHEMA = {"type": "object", "properties": {
    "dishes": enum_list(DISHES), "adjectives": enum_list(ADJECTIVES), "aspects": enum_list(ASPECTS),
    "exclude": enum_list(DISHES + ADJECTIVES), "districts": enum_list(DISTRICTS),
    "time_tags": enum_list(TIME_TAGS), "price_max": {"type": ["number", "null"]},
    "sort_by": {"type": "string", "enum": SORTS}, "clarify": {"type": "boolean"}},
    "required": ["dishes", "adjectives", "aspects", "exclude", "districts", "time_tags", "price_max", "sort_by", "clarify"]}

queries = [json.loads(l)["query"] for l in open("/kaggle/working/pairs.jsonl", encoding="utf-8")]
# Thêm các câu khó mà rule parser đang đọc kém, cùng một ít câu không phải tìm quán (để học clarify=true).
labels = ask([[{"role": "system", "content": PROMPT}, {"role": "user", "content": q}] for q in queries],
             PARSER_SCHEMA, temperature=0.0, max_tokens=256)
with open("/kaggle/working/parser.jsonl", "w", encoding="utf-8") as f:
    for q, y in zip(queries, labels):
        y["rule_confidence"] = parse_intent(q).confidence   # để sau này so model với luật
        f.write(json.dumps({"query": q, **y}, ensure_ascii=False) + "\n")
```

### 2c. Nhãn ABSA bằng LLM (silver label)

```python
ABSA_SCHEMA = {"type": "object", "properties": {a: {"type": "string", "enum": ["none", "positive", "negative", "neutral"]}
               for a in ASPECTS}, "required": ASPECTS}
ABSA_PROMPT = ("Với review quán ăn dưới đây, cho biết review nói gì về từng khía cạnh: food (món ăn), price (giá), "
               "service (phục vụ), space (không gian), hygiene (vệ sinh), parking (chỗ để xe). "
               "Khía cạnh không được nhắc tới thì ghi 'none'.")
texts = [t for ts in reviews.values() for t in ts]
absa = ask([[{"role": "system", "content": ABSA_PROMPT}, {"role": "user", "content": t}] for t in texts],
           ABSA_SCHEMA, temperature=0.0, max_tokens=128)
with open("/kaggle/working/absa.jsonl", "w", encoding="utf-8") as f:
    for t, y in zip(texts, absa):
        f.write(json.dumps({"text": t, **y}, ensure_ascii=False) + "\n")
```

Sau đó **tự kiểm tra tay khoảng 300 review**. Phần này dùng làm tập test thật; nếu đo trên nhãn của chính LLM thì chỉ là đo độ giống LLM.

Cuối notebook, lưu ba file `.jsonl` thành một Kaggle Dataset (*Save Version → Output*) để các notebook sau dùng làm input.

---

## 3. Fine-tune embedding

```python
import random, pandas as pd, torch
import text_utils as tu
from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments, losses)
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.util import mine_hard_negatives
from sentence_transformers.evaluation import InformationRetrievalEvaluator

BASE = "AITeamVN/Vietnamese_Embedding_v2"       # model thắng ở notebook 1
QP, PP = "", ""                                  # prefix của model đó (Qwen3: QP = "Instruct: ...\nQuery:")
passage = {r.id: PP + f"{r.name}. {r.tags}".strip(". ") for r in frame.itertuples()}   # giống data_store

pairs = pd.read_json("/kaggle/input/<dataset>/pairs.jsonl", lines=True)
# Thêm bản không dấu để model học được "bun bo q3".
pairs = pd.concat([pairs, pairs.assign(query=pairs["query"].map(tu.fold))]).drop_duplicates()

# Tách theo QUÁN chứ không theo câu: quán ở tập eval thì model chưa thấy lúc train.
ids = sorted(pairs["id"].unique()); random.Random(0).shuffle(ids)
held = set(ids[: len(ids) // 10])
train = pairs[~pairs["id"].isin(held)]
dev = pairs[pairs["id"].isin(held)]

model = SentenceTransformer(BASE)
train_ds = Dataset.from_dict({"anchor": [QP + q for q in train["query"]],
                              "positive": [passage[i] for i in train["id"]]})
# Hard negative lấy ở hạng 10-50, không lấy hạng đầu: các hạng đầu thường cũng là quán đúng (false negative).
train_ds = mine_hard_negatives(train_ds, model, num_negatives=1, range_min=10, range_max=50,
                               sampling_strategy="random", batch_size=128, output_format="triplet")

evaluator = InformationRetrievalEvaluator(
    queries={str(k): QP + q for k, q in enumerate(dev["query"])},
    corpus={i: p for i, p in passage.items()},
    relevant_docs={str(k): {i} for k, i in enumerate(dev["id"])},
    name="dev", show_progress_bar=False)

args = SentenceTransformerTrainingArguments(
    output_dir="/kaggle/working/emb-ft", num_train_epochs=1, per_device_train_batch_size=64,
    learning_rate=2e-5, warmup_ratio=0.1, fp16=True,                 # T4: fp16, không dùng bf16
    batch_sampler=BatchSamplers.NO_DUPLICATES, eval_strategy="steps", eval_steps=200,
    save_strategy="steps", save_steps=200, save_total_limit=2, load_best_model_at_end=True,
    metric_for_best_model="eval_dev_cosine_ndcg@10", report_to="none")
# bge-m3 khá nặng: CachedMNRL cho phép batch lớn mà không tràn 16 GB.
loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=16)
SentenceTransformerTrainer(model=model, args=args, train_dataset=train_ds, loss=loss,
                           evaluator=evaluator).train()
print(evaluator(model))
model.push_to_hub("ddanh2436/vietnomnom-embedding", private=True)
```

Sau khi train xong, chạy lại `bench()` của notebook 1 với `EMBEDDING_MODEL=ddanh2436/vietnomnom-embedding` để có
`.npz` mới và so với baseline. Nếu model để private, Space cần secret `HF_TOKEN` (chỉ quyền read) để tải.

**Lưu ý false negative:** câu "phở quận 3" đúng với mọi quán phở ở Quận 3, nhưng nếu trong cùng một batch có quán phở Quận 3 khác
thì MNRL vẫn coi quán đó là sai. Khi loss trên tập dev không giảm thêm, hãy lọc các negative có chung món và chung quận
với câu truy vấn (dùng `parse_intent(q).dishes` và cột `district`).

---

## 4. Fine-tune reranker

Reranker chấm lại khoảng 20–30 quán đầu của pipeline hiện tại. Trên CPU, chi phí phụ thuộc vào kích thước model nhân với số cặp cần chấm:

| Model | Kích thước | Ghi chú |
| --- | --- | --- |
| `Horizon-Labs/multilingual-reranker-small` | 141M | nhẹ nhất, distill từ Qwen3-Reranker-4B |
| `Alibaba-NLP/gte-multilingual-reranker-base` | 306M | `trust_remote_code=True` |
| ViRanker (bge-m3) | 568M | chuyên tiếng Việt, nhưng có thể quá chậm cho 30 cặp trên 2 vCPU |
| `Qwen/Qwen3-Reranker-0.6B` | 600M | là LLM decoder nên chạy CPU chậm; không khuyến nghị cho Space free |

```python
from sentence_transformers.cross_encoder import (CrossEncoder, CrossEncoderTrainer,
                                                 CrossEncoderTrainingArguments, losses as ce_losses)
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator

embedder = SentenceTransformer("ddanh2436/vietnomnom-embedding")      # dùng model ở bước 3 để mine
pair_ds = Dataset.from_dict({"query": list(train["query"]), "answer": [passage[i] for i in train["id"]]})
labeled = mine_hard_negatives(pair_ds, embedder, num_negatives=4, range_min=10, range_max=60,
                              sampling_strategy="random", output_format="labeled-pair")

reranker = CrossEncoder("Horizon-Labs/multilingual-reranker-small", num_labels=1, max_length=128)
args = CrossEncoderTrainingArguments(output_dir="/kaggle/working/rr-ft", num_train_epochs=1,
                                     per_device_train_batch_size=64, learning_rate=2e-5, warmup_ratio=0.1,
                                     fp16=True, report_to="none")
CrossEncoderTrainer(model=reranker, args=args, train_dataset=labeled,
                    loss=ce_losses.BinaryCrossEntropyLoss(reranker, pos_weight=torch.tensor(4.0))).train()
reranker.push_to_hub("ddanh2436/vietnomnom-reranker", private=True)
```

Phải đo độ trễ trên CPU với 30 cặp (`torch.set_num_threads(2)`) **trước khi** tích hợp. Ngân sách hợp lý là dưới 300 ms.
Phần tích hợp vào `search.py` (chấm lại top 30 rồi trộn điểm với `relevance`) chỉ nên viết khi model đã vượt baseline trên
`benchmark.py`.

---

## 5. Parser cục bộ thay Gemini

`llm_parser.refine()` chỉ nhận một dict gồm các giá trị **thuộc danh sách cố định**. Vì vậy chỗ thay thế đúng là hàm `_ask()`,
và chỉ cần một encoder **phân loại đa nhãn**, không cần LLM sinh văn bản. Model này chạy khoảng 15–30 ms trên CPU và không bị giới hạn quota.

Phần giá tiền vẫn để luật đọc như hiện tại, vì số tiền là thứ luật đọc chính xác nhất.

```python
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments)
from llm_parser import SORTS
import numpy as np

rows = pd.read_json("/kaggle/input/<dataset>/parser.jsonl", lines=True)
FIELDS = ["dishes", "adjectives", "aspects", "exclude", "districts", "time_tags"]
LABELS = [f"{f}:{v}" for f in FIELDS for v in sorted({v for vs in rows[f] for v in vs})]
LABELS += [f"sort_by:{s}" for s in SORTS] + ["clarify"]
index = {l: i for i, l in enumerate(LABELS)}

def target(r):
    y = np.zeros(len(LABELS), dtype="float32")
    for f in FIELDS:
        for v in r[f]: y[index[f"{f}:{v}"]] = 1
    y[index[f"sort_by:{r['sort_by']}"]] = 1
    y[index["clarify"]] = float(r["clarify"])
    return y

BASE = "uitnlp/visobert"          # không cần tách từ; xem ghi chú license ở cuối
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=len(LABELS), problem_type="multi_label_classification")
ds = Dataset.from_dict({"text": list(rows["query"]) + list(rows["query"].map(tu.fold)),
                        "labels": [target(r) for _, r in rows.iterrows()] * 2})
ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=64), batched=True).train_test_split(0.1, seed=0)
Trainer(model=model, args=TrainingArguments("/kaggle/working/parser-ft", num_train_epochs=4,
        per_device_train_batch_size=64, learning_rate=5e-5, fp16=True, report_to="none"),
        train_dataset=ds["train"], eval_dataset=ds["test"], processing_class=tok).train()
model.config.id2label = dict(enumerate(LABELS))
model.push_to_hub("ddanh2436/vietnomnom-parser", private=True); tok.push_to_hub("ddanh2436/vietnomnom-parser", private=True)
```

Ngưỡng 0.5 cho mỗi nhãn là điểm khởi đầu. Hãy chỉnh ngưỡng trên tập dev rồi chạy `benchmark.py`, chủ yếu xem nhóm `hard`.
Nhóm này hiện chỉ đạt 2/10 khi không có Gemini, nên đây là thước đo trực tiếp cho parser mới.

---

## 6. ABSA (6 khía cạnh)

Phần này thay việc đếm từ khoá trong `aspect_index.py`. Có hai nguồn dữ liệu:

- **VLSP 2018 ABSA, mảng nhà hàng** (xin quyền truy cập ở vlsp.org.vn/resources). Bộ này có FOOD, PRICE, SERVICE, AMBIENCE, LOCATION,
  nhưng **không có hygiene và parking**.
- **`absa.jsonl` ở bước 2c**: bổ sung hai khía cạnh còn thiếu ở trên, và lấy đúng văn phong review trên web của bạn.

Model có 6 đầu ra, mỗi đầu phân loại 4 nhãn (`none / positive / negative / neutral`) trên nền ViSoBERT, dùng loss là tổng
cross-entropy của 6 đầu. Đánh giá trên khoảng 300 review đã kiểm tra tay. Khi tích hợp, chỉ cần thay chỗ `aspect_index.build()` gán
`positive_ratio` và `mentions`. Ranker và trang chi tiết quán đã đọc sẵn hai trường này.

---

## 7. Sentiment 4 nhãn cho từng review

**Vì sao cần:** so nhãn của model hiện tại (`5CD-AI/Vietnamese-Sentiment-visobert`) với điểm chính người viết chấm, trên 26.529 review:

| Điểm người viết chấm | POS | NEU | NEG |
| --- | --- | --- | --- |
| ≥ 8 | 96% | 3% | 1% |
| 5–7 | 67% | 12% | 21% |
| ≤ 4 | 6% | 2% | 92% |

Model đúng ở hai đầu nhưng sai ở khoảng giữa, vì nó **không có nhãn "trái chiều"**: "ngon nhưng giá chát" bị ép thành POS hoặc NEG.
Model mới có 4 nhãn: `POS` (khen là chính), `NEG` (chê là chính), `MIX` (vừa khen vừa chê), `NEU` (kể lại, hỏi, không bày tỏ ý kiến).

### 7a. Bộ test thật (chấm tay, làm trên máy)

```bash
python sentiment_sheet.py      # tạo sentiment_to_grade.json: 400 review, nặng về nhóm điểm 5-7
```

Mở file, đổi `"label": null` thành `"POS"`, `"NEG"`, `"MIX"` hoặc `"NEU"`. **Chấm theo nội dung, không theo điểm.** Chấm xong,
upload file này thành một **Kaggle Dataset private** tên `vietnomnom-gold`. File đã được đưa vào `.gitignore` vì chứa nguyên văn review Foody.
Đây là đề thi: notebook sẽ loại các review này khỏi tập train.

### 7b. Nhãn silver bằng LLM (chạy trong môi trường llama.cpp của notebook 2)

Không tự chấm tay 26k review được, nên để Gemma gán nhãn, rồi dùng điểm của người viết để loại các nhãn vô lý.
**Đo trước trên bộ gold:** chạy `ask` cho 400 review đã chấm tay và tính tỉ lệ khớp. Khớp từ khoảng 80% trở lên mới
đáng chạy cả 26k; thấp hơn thì sửa prompt trước, vì nhãn silver sai bao nhiêu thì model học sai bấy nhiêu.

```python
import json, os
from pymongo import MongoClient
db = MongoClient(os.environ["MONGO_URI"])["VietNomNom"]
gold = {r["text"] for r in json.load(open("/kaggle/input/vietnomnom-gold/sentiment_to_grade.json"))["reviews"]}
rows = []
for r in db["reviews"].find({}, {"noiDung": 1, "diemReview": 1, "urlGoc": 1}):
    text = " ".join((r.get("noiDung") or "").split())
    if len(text) >= 20 and text not in gold:            # đề thi không được lọt vào tập train
        # Thiếu điểm thì coi như 5: không đủ căn cứ để loại nhãn nào.
        rows.append({"text": text, "score": float(r.get("diemReview") or 5), "place": r.get("urlGoc")})

SENT_SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": ["POS", "NEG", "MIX", "NEU"]}},
               "required": ["label"]}
SENT_PROMPT = ("Phân loại cảm xúc của review quán ăn. POS: khen là chính. NEG: chê là chính. "
               "MIX: vừa khen vừa chê rõ ràng, ví dụ 'ngon nhưng giá chát'. NEU: chỉ kể lại, hỏi, không bày tỏ ý kiến. "
               "Review trích lời chê của người khác để phản bác thì vẫn tính theo ý của người viết.")
# Đo trước trên 400 review chấm tay: khớp dưới ~80% thì sửa prompt, đừng chạy cả 26k.
gold_rows = [r for r in json.load(open("/kaggle/input/vietnomnom-gold/sentiment_to_grade.json"))["reviews"] if r["label"]]
pred = ask([[{"role": "system", "content": SENT_PROMPT}, {"role": "user", "content": r["text"][:2000]}] for r in gold_rows],
           SENT_SCHEMA, temperature=0.0, max_tokens=16)
print("khớp nhãn tay:", round(sum(p["label"] == r["label"] for p, r in zip(pred, gold_rows)) / len(gold_rows), 3))

out = ask([[{"role": "system", "content": SENT_PROMPT}, {"role": "user", "content": r["text"][:2000]}] for r in rows],
          SENT_SCHEMA, temperature=0.0, max_tokens=16)
for r, y in zip(rows, out):
    r["label"] = y["label"]

# LLM nói POS mà người viết chấm <= 4, hoặc nói NEG mà chấm >= 8.5: một trong hai sai, bỏ cho chắc.
keep = [r for r in rows if not (r["label"] == "POS" and r["score"] <= 4 or r["label"] == "NEG" and r["score"] >= 8.5)]
with open("/kaggle/working/sentiment_silver.jsonl", "w", encoding="utf-8") as f:
    for r in keep:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), "->", len(keep))
```

Khoảng 26k review × 300 token, mất 1–2 giờ trên T4. Lưu `sentiment_silver.jsonl` thành Kaggle Dataset để dùng ở bước sau.

### 7c. Fine-tune (notebook riêng, không cần llama.cpp)

Khởi đầu từ chính model sentiment đang dùng, vì nó đã biết khen/chê, rồi thay đầu ra 3 nhãn bằng 4 nhãn.

```python
import json, random, numpy as np, torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

LABELS = ["NEG", "NEU", "POS", "MIX"]               # 3 nhãn đầu giữ thứ tự cũ (LABEL_0/1/2)
L2I = {l: i for i, l in enumerate(LABELS)}
BASE = "5CD-AI/Vietnamese-Sentiment-visobert"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=4, id2label=dict(enumerate(LABELS)), label2id=L2I, ignore_mismatched_sizes=True)

rows = [json.loads(l) for l in open("/kaggle/input/vietnomnom-silver/sentiment_silver.jsonl", encoding="utf-8")]
# Tách theo QUÁN: review của cùng một quán hay giống nhau, tách theo câu thì điểm val bị thổi phồng.
places = sorted({r["place"] for r in rows}); random.Random(0).shuffle(places)
val_places = set(places[: len(places) // 10])

def encode(r):
    # Review dài: giữ 128 token đầu và 126 token cuối, vì kết luận ("nói chung là...") thường nằm ở cuối.
    ids = tok(r["text"], add_special_tokens=False)["input_ids"]
    if len(ids) > 254:
        ids = ids[:128] + ids[-126:]
    ids = [tok.cls_token_id] + ids + [tok.sep_token_id]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": L2I[r["label"]]}

train = [encode(r) for r in rows if r["place"] not in val_places]
val = [encode(r) for r in rows if r["place"] in val_places]
counts = Counter(x["labels"] for x in train)
weights = torch.tensor([len(train) / (4 * counts[i]) for i in range(4)], dtype=torch.float)
print(counts)  # MIX và NEU sẽ ít hơn POS nhiều: trọng số loss bù lại

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        out = model(**inputs)
        loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=weights.to(out.logits.device))
        return (loss, out) if return_outputs else loss

def collate(batch):
    n = max(len(x["input_ids"]) for x in batch)
    pad = lambda k, v: [x[k] + [v] * (n - len(x[k])) for x in batch]
    return {"input_ids": torch.tensor(pad("input_ids", tok.pad_token_id)),
            "attention_mask": torch.tensor(pad("attention_mask", 0)),
            "labels": torch.tensor([x["labels"] for x in batch])}

args = TrainingArguments("/kaggle/working/sent4", per_device_train_batch_size=32, per_device_eval_batch_size=64,
                         learning_rate=2e-5, num_train_epochs=3, warmup_ratio=0.1, weight_decay=0.01,
                         fp16=True, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
                         metric_for_best_model="eval_loss", save_total_limit=1, report_to=[])
WeightedTrainer(model=model, args=args, train_dataset=train, eval_dataset=val, data_collator=collate).train()
```

### 7d. Đo trên bộ test thật

```python
from sklearn.metrics import classification_report, confusion_matrix, f1_score
gold_rows = [r for r in json.load(open("/kaggle/input/vietnomnom-gold/sentiment_to_grade.json"))["reviews"] if r["label"]]

def predict(m, texts):
    m.eval(); preds = []
    for i in range(0, len(texts), 64):
        b = collate([encode({"text": t, "label": "POS"}) for t in texts[i:i + 64]])
        with torch.no_grad():
            logits = m(input_ids=b["input_ids"].to(m.device), attention_mask=b["attention_mask"].to(m.device)).logits
        preds += [m.config.id2label[j] for j in logits.argmax(-1).tolist()]
    return preds

y = [r["label"] for r in gold_rows]
new = predict(model, [r["text"] for r in gold_rows])
old_model = AutoModelForSequenceClassification.from_pretrained(BASE).to(model.device)
old = [{"LABEL_0": "NEG", "LABEL_1": "NEU", "LABEL_2": "POS"}.get(p, p) for p in predict(old_model, [r["text"] for r in gold_rows])]
for name, p in [("cũ (3 nhãn)", old), ("mới (4 nhãn)", new)]:
    print(name, "macro-F1:", round(f1_score(y, p, average="macro", labels=LABELS), 3))
print(classification_report(y, new, labels=LABELS, digits=3))
print(confusion_matrix(y, new, labels=LABELS))
```

**Chỉ giữ model mới khi** macro-F1 cao hơn model cũ và F1 của `MIX` từ khoảng 0.6 trở lên. Nếu `MIX` thấp, xem ma trận nhầm lẫn:
MIX hay bị nhầm thành POS thì thêm ví dụ MIX (ví dụ chạy lại 7b với `temperature=0.0` riêng cho nhóm điểm 5–7), chứ đừng hạ ngưỡng.

### 7e. Đẩy lên Hub và báo lại

```python
model.push_to_hub("sirin2412/vietnomnom-sentiment-4c", private=True)
tok.push_to_hub("sirin2412/vietnomnom-sentiment-4c", private=True)
```

Sau đó gửi tên model cùng kết quả 7d. Phần tích hợp gồm: đặt `SENTIMENT_MODEL` và `HF_TOKEN` (để tải model private) trên Space,
cho `sentiment.py` hiểu nhãn `MIX`, hiển thị "Trái chiều" ở trang quán, chạy lại nhãn cho 26k review và chạy lại aspect index.

---

## Lưu ý

- **License:** ViSoBERT (cả model sentiment đang dùng) chỉ được phép dùng cho nghiên cứu. Với đồ án thì không sao, nhưng nếu
  thương mại hoá thì nên đổi sang base có license mở, như `FacebookAI/xlm-roberta-base` (MIT) hoặc model Apache 2.0.
  Gemma 4 dùng Apache 2.0.
- **Dữ liệu crawl từ Foody:** chỉ đẩy model lên Hub ở chế độ private, không public dataset `pairs.jsonl` hay review.
- **Không đo trên dữ liệu đã dùng để train:** `search_eval.json` và `search_qrels.json` là đề thi. Bước 2a đã lọc trùng; khi thêm case mới
  vào bộ đo, đừng chép câu từ `pairs.jsonl` sang.
