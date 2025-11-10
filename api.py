import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from pathlib import Path
import os

# -----------------------------------------------------------------
# BƯỚC 1: ĐỊNH NGHĨA "HỢP ĐỒNG" (PYDANTIC SCHEMAS)
# -----------------------------------------------------------------
# (Không thay đổi)
class RecommendRequest(BaseModel):
    query: str 
    candidate_ids: List[int]

class TasteScore(BaseModel):
    id: int
    name: str 
    tags: str  
    S_taste: float

class RecommendResponse(BaseModel):
    sort_by: str 
    scores: List[TasteScore] 

class SentimentRequest(BaseModel):
    review: str

class SentimentResponse(BaseModel):
    label: str
    score: float

# -----------------------------------------------------------------
# BƯỚC 2: KHỞI TẠO APP VÀ TẢI MÔ HÌNH (Tải 1 lần)
# -----------------------------------------------------------------
print("--- KHỞI ĐỘNG SERVER AI (100% TỰ HOST - VFINAL) ---")
app = FastAPI(title="Smart Tourism AI Toolbox API (Self-Hosted)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- (Biến Global cho Model, sẽ được tải trong 'startup' event) ---
df = None
tfidf_vectorizer = None
tag_matrix = None
sentiment_pipeline = None
semantic_tokenizer = None
semantic_model = None
candidate_vectors = None

# --- (Tải "Lớp Tri Thức" và "Tags Sạch" - Không thay đổi) ---
KNOWLEDGE_BASE_RULES = {
    'huế': 'miền trung', 'đà nẵng': 'miền trung', 'hội an': 'miền trung',
    'hà nội': 'bắc', 'nam định': 'bắc', 'sài gòn': 'miền nam', 'miền tây': 'miền tây',
    'lãng mạn': 'hẹn hò', 'sang chảnh': 'sang trọng', 'đắt tiền': 'sang trọng',
    'thoải mái': 'yên tĩnh', 'nhanh gọn': 'nhanh', 'tụ tập': 'nhậu bạn bè', 'bạn bè': 'nhậu bạn bè',
    'đồng quê': 'cơm việt truyền thống',
    'gần': 'sort_distance', 'rẻ': 'sort_price', 'ngon': 'sort_rating',
    'tốt nhất': 'sort_rating', 'đánh giá cao': 'sort_rating'
}
candidate_tags = [
    'bún bò', 'phở', 'cơm tấm', 'pizza', 'gà rán', 'bánh xèo', 'mì quảng', 'bún đậu', 'ốc', 'hải sản', 'sushi', 'lẩu', 'bò 7 món', 'hủ tiếu', 'đồ nướng', 'bánh mì', 'cơm việt', 'lẩu dê', 'phá lấu', 'steak', 'ramen', 'dimsum', 'lẩu trung hoa', 'bánh canh cua', 'mì ý', 'bia thủ công', 'chả cá', 'cuốn', 'bò né', 'cơm niêu', 'cơm lam', 'cơm gà', 'bánh cuốn', 'lẩu nấm', 'cháo sườn', 'mì vằn thắn', 'vịt quay', 'bánh bột lọc', 'bò kho', 'bún mắm', 'bánh tráng nướng', 'nem nướng', 'lẩu cá kèo', 'bún riêu', 'mì vịt tiềm', 'hủ tiếu mực', 'gà nướng', 'cơm văn phòng', 'chè thái', 'ăn vặt', 'tráng miệng', 'trà sữa', 'bingsu', 'kem', 'xiên que', 'đậu hũ', 'tàu hủ', 'bánh ngọt', 'xôi gà', 'xôi bắp', 'king roti', 'cay', 'ngọt', 'chay', 'mắm tôm', 'phô mai', 'xối mỡ', '7 cấp độ', 'chua', 'miền trung', 'bắc', 'hàn quốc', 'ý', 'nhật', 'trung hoa', 'âu', 'miền nam', 'miền tây', 'nam vang', 'á', 'vỉa hè', 'sang trọng', 'yên tĩnh', 'hẹn hò', 'truyền thống', 'nhậu', 'đêm', 'nhanh', 'buffet', 'mang đi', 'làm việc', 'gia đình', 'bạn bè', 'bình dân', 'dịch vụ', 'sáng', 'trưa', 'băng chuyền', 'điểm tâm', 'chuỗi', 'cà phê vợt'
]

# --- (Hàm get_semantic_vector - Không thay đổi) ---
def get_semantic_vector(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:,0,:].numpy()

# --- SỰ KIỆN STARTUP (Tải Model khi Uvicorn khởi động) ---
@app.on_event("startup")
async def startup_event():
    global df, tfidf_vectorizer, tag_matrix, sentiment_pipeline
    global semantic_tokenizer, semantic_model, candidate_vectors

    # === Tải AI 1: GĐ 3A (Scikit-learn) ===
    print("Đang tải AI GĐ 3A (Scikit-learn - So khớp)...")
    try:
        BASE_DIR = Path(__file__).resolve().parent
        CSV_PATH = os.path.normpath(BASE_DIR / 'restaurants.csv') # (Đã sửa tên file)
        print(f"Đang đọc file CSV từ: {CSV_PATH}")
        
        df = pd.read_csv(CSV_PATH)
        
        # --- (SỬA LỖI: Ép kiểu ID về SỐ NGUYÊN (INT)) ---
        df['id'] = pd.to_numeric(df['id'], errors='coerce') 
        df = df.dropna(subset=['id'])
        df['id'] = df['id'].astype(int) # (Sửa lỗi: Ép về INT)
        # --- (HẾT SỬA LỖI ID) ---

        df['tags'] = df['tags'].fillna('')
        
        tfidf_vectorizer = TfidfVectorizer()
        tag_matrix = tfidf_vectorizer.fit_transform(df['tags'])
        print("AI GĐ 3A (Scikit-learn) đã sẵn sàng!")
    except FileNotFoundError:
        print(f"LỖI KHÔNG TÌM THẤY FILE: 'restaurants.csv' tại {BASE_DIR}")
        exit()
    except Exception as e:
        print(f"LỖI GĐ 3A: {e}")

    # === (Tải AI 2 (Sentiment) và AI 3 (PhoBERT) - Không thay đổi) ===
    print("Đang tải AI GĐ 3B (5CD-AI Sentiment)...")
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model="5CD-AI/Vietnamese-Sentiment-visobert")
        print("AI GĐ 3B (5CD-AI Sentiment) đã sẵn sàng!")
    except Exception as e:
        print(f"LỖI GĐ 3B (Sentiment): {e}")

    print("Đang tải AI Lớp 3 (PhoBERT Semantic Search)...")
    try:
        semantic_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        semantic_model = AutoModel.from_pretrained("vinai/phobert-base-v2")
        print("AI Lớp 3 (PhoBERT) đã sẵn sàng!")
        
        print("Đang tạo vector cho các tags (Semantic Search)...")
        candidate_vectors = get_semantic_vector(candidate_tags, semantic_tokenizer, semantic_model)
        candidate_vectors = candidate_vectors.reshape(len(candidate_tags), -1) 
        print(f"Vector tags (Lớp 3) đã sẵn sàng! Shape: {candidate_vectors.shape}")
    except Exception as e:
        print(f"LỖI LỚP 3 (PhoBERT): {e}")

    print("--- SERVER AI ĐÃ SẴN SÀNG CHỜ REQUEST ---")

# -----------------------------------------------------------------
# BƯỚC 3: TẠO CÁC "CÁNH CỬA" API (ENDPOINTS)
# -----------------------------------------------------------------

@app.post("/recommend", response_model=RecommendResponse)
async def handle_recommendation(request_data: RecommendRequest):
    if df is None or tfidf_vectorizer is None:
        raise HTTPException(status_code=503, detail="AI models not loaded yet")
        
    # 1. Nhận query và candidate_ids (là List[INT])
    query_lower = request_data.query.lower()
    candidate_ids_int = request_data.candidate_ids
    print(f"\n[Request /recommend] Nhận query: '{query_lower}'")
    
    # 2. CHẠY "PIPELINE 3 LỚP" (Xử lý Query)
    # (Toàn bộ code Pipeline 3 Lớp giữ nguyên y hệt, không cần sửa)
    processed_query = " " + query_lower + " "
    sort_by = "taste" 
    for keyword, replacement in KNOWLEDGE_BASE_RULES.items():
        if f" {keyword} " in processed_query:
            if "sort_" in replacement: 
                sort_by = replacement.replace("sort_", "") 
                processed_query = processed_query.replace(f" {keyword} ", " ")
            else:
                processed_query = processed_query.replace(f" {keyword} ", f" {replacement} ")
    print(f"[Lớp 1] Query đã qua Lớp Tri Thức: '{processed_query}', Sort_by: '{sort_by}'")
    
    extracted_tags = []
    for tag in candidate_tags:
        if f" {tag.lower()} " in processed_query: 
            extracted_tags.append(tag)
            
    if not extracted_tags and semantic_model is not None:
        print("Lối đi nhanh thất bại. Chuyển sang Lớp 3 (Semantic Search)...")
        threshold = 0.6 
        try:
            query_vector = get_semantic_vector(query_lower, semantic_tokenizer, semantic_model)
            semantic_scores = cosine_similarity(query_vector, candidate_vectors).flatten()
            for i, score in enumerate(semantic_scores):
                if score > threshold:
                    extracted_tags.append(candidate_tags[i])
        except Exception as e:
            print(f"LỖI LỚP 3 (Semantic): {e}")
    else:
        print("Tìm thấy tag (Lớp 1/Lớp 2). Bỏ qua Lớp 3.")
    
    final_query = " ".join(extracted_tags)
    if not final_query: 
        final_query = query_lower
    print(f"[Pipeline] Query đã xử lý (Cuối cùng): '{final_query}'")

    # 3. CHẠY GĐ 3A (TÍNH ĐIỂM "TASTE")
    
    # --- (SỬA LỖI: So sánh INT vs INT) ---
    # `df['id']` (là INT) so với `candidate_ids_int` (là INT)
    filtered_df = df[df['id'].isin(candidate_ids_int)]
    # --- (HẾT SỬA LỖI) ---
    
    if filtered_df.empty:
        print("DEBUG: filtered_df rỗng. (Không tìm thấy ID ứng viên trong 'df')")
        return {"sort_by": sort_by, "scores": []} 

    query_vector_3a = tfidf_vectorizer.transform([final_query])
    filtered_tag_matrix = tfidf_vectorizer.transform(filtered_df['tags'])
    cosine_scores_3a = cosine_similarity(query_vector_3a, filtered_tag_matrix).flatten()

    # --- (Phần trả về 'name', 'tags' giữ nguyên) ---
    scores_list = []
    for i, index in enumerate(filtered_df.index):
        restaurant_data = filtered_df.loc[index]
        scores_list.append({
            "id": int(restaurant_data['id']),
            "name": str(restaurant_data['name']),
            "tags": str(restaurant_data['tags']),
            "S_taste": float(cosine_scores_3a[i])
        })
        
    print(f"Trả về {len(scores_list)} điểm 'Taste' và 'sort_by: {sort_by}'.")
    return {"sort_by": sort_by, "scores": scores_list}


# --- Endpoint 2: Phân tích Review (Sentiment) ---
@app.post("/sentiment", response_model=SentimentResponse)
async def handle_sentiment(request_data: SentimentRequest):
    # (Code này giữ nguyên)
    if sentiment_pipeline is None:
        raise HTTPException(status_code=503, detail="Sentiment model not loaded")
    review_text = request_data.review
    # ... (code còn lại giữ nguyên) ...
    try:
        result = sentiment_pipeline(review_text)
        return result[0] 
    except Exception as e:
        print(f"LỖI GĐ 3B (Sentiment): {e}")
        raise HTTPException(status_code=500, detail="AI model failed")

# -----------------------------------------------------------------
# BƯỚC 4: CHẠY SERVER (Bằng lệnh Terminal)
# -----------------------------------------------------------------
# (Xóa 4 dòng if __name__ == "__main__": ...)
# (Hãy chạy bằng lệnh: py -m uvicorn api:app --reload)