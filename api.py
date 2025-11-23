import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import os
import math
import re

# -----------------------------------------------------------------
# 1. CẤU HÌNH & MODELS
# -----------------------------------------------------------------

class RecommendRequest(BaseModel):
    query: str 
    candidate_ids: Optional[List[int]] = []
    user_gps: Optional[List[float]] = None
    city_filter: Optional[str] = None

class TasteScore(BaseModel):
    id: int
    name: str 
    tags: str  
    S_taste: float
    distance_km: float
    price: int

class RecommendResponse(BaseModel):
    sort_by: str 
    scores: List[TasteScore] 

class SentimentRequest(BaseModel):
    review: str

class SentimentResponse(BaseModel):
    label: str
    score: float

# Hàm tính khoảng cách (Haversine)
def calculate_distance(lat1, lon1, lat2, lon2):
    if lat1 is None or lat2 is None or pd.isna(lat1) or pd.isna(lat2): return 999.0
    try:
        R = 6371 
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        a = math.sin(dLat/2) * math.sin(dLat/2) + \
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
            math.sin(dLon/2) * math.sin(dLon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    except: return 999.0

def get_semantic_vector(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad(): outputs = model(**inputs)
    return outputs.last_hidden_state[:,0,:].numpy()

print("--- KHỞI ĐỘNG SERVER AI (V-FINAL-RADIUS) ---")
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

df = None
tfidf_vectorizer = None
tag_matrix = None
sentiment_pipeline = None
semantic_tokenizer = None
semantic_model = None
candidate_vectors = None

# --- KNOWLEDGE BASE ---
TAG_KNOWLEDGE_BASE = {
    'lãng mạn': 'hẹn hò', 'sang chảnh': 'sang trọng', 'đắt tiền': 'sang trọng',
    'thoải mái': 'yên tĩnh', 'nhanh gọn': 'nhanh', 'tụ tập': 'nhậu bạn bè', 
    'đồng quê': 'cơm việt truyền thống', 'đặc sản huế': 'bún bò miền trung', 
    'đặc sản hà nội': 'phở bún chả bắc', 'đặc sản sài gòn': 'cơm tấm miền nam'
}
SORT_KNOWLEDGE_BASE = {
    'gần': 'distance', 'rẻ': 'price', 'ngon': 'rating', 'tốt nhất': 'rating'
}
LOCATION_NAMES = {
    'hà nội': ['Hà Nội'], 'sài gòn': ['TPHCM'], 'tp.hcm': ['TPHCM'], 
    'hồ chí minh': ['TPHCM'], 'quận 1': ['Quận 1']
}
LOCATION_TRIGGERS = ['ở', 'tại', 'đến', 'thuộc']
candidate_tags = [
    'bún bò', 'phở', 'cơm tấm', 'pizza', 'gà rán', 'bánh xèo', 'mì quảng', 'bún đậu', 'ốc', 'hải sản', 'sushi', 'lẩu', 'bò', 'hủ tiếu', 'nướng', 'bánh mì', 'cơm việt', 'dê', 'phá lấu', 'steak', 'ramen', 'dimsum', 'bánh canh', 'cua', 'mì ý', 'bia thủ công', 'chả cá', 'bingsu', 'xôi', 'cuốn', 'bò né', 'kem', 'xiên que', 'cơm niêu', 'cà phê vợt', 'cơm lam', 'đậu hũ', 'cay', 'ngọt', 'chay', 'mắm tôm', 'phô mai', 'miền trung', 'bắc', 'hàn quốc', 'ý', 'nhật', 'trung hoa', 'âu', 'miền nam', 'miền tây', 'vỉa hè', 'sang trọng', 'yên tĩnh', 'hẹn hò', 'truyền thống', 'nhậu', 'đêm', 'nhanh', 'buffet', 'mang đi', 'làm việc', 'gia đình', 'bình dân', 'dịch vụ', 'sáng', 'trưa', 'chè', 'ăn vặt', 'trà sữa', 'bánh', 'cơm', 'mì', 'gà', 'cá'
]

@app.on_event("startup")
async def startup_event():
    global df, tfidf_vectorizer, tag_matrix, sentiment_pipeline, semantic_tokenizer, semantic_model, candidate_vectors
    try:
        BASE_DIR = Path(__file__).resolve().parent
        CSV_PATH = os.path.normpath(BASE_DIR / 'restaurants.csv')
        print(f"Đọc CSV: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
        df['id'] = pd.to_numeric(df['id'], errors='coerce')
        df = df.dropna(subset=['id'])
        df['id'] = df['id'].astype(int)
        df['tags'] = df['tags'].fillna('')
        df['district'] = df['district'].fillna('').astype(str)
        if 'lat' not in df.columns: df['lat'] = 0.0
        if 'lon' not in df.columns: df['lon'] = 0.0
        if 'price' not in df.columns: df['price'] = 0
        
        tfidf_vectorizer = TfidfVectorizer()
        tag_matrix = tfidf_vectorizer.fit_transform(df['tags'])
        print(f"AI GĐ 3A Sẵn sàng! (Data: {len(df)} quán)")
    except Exception as e: print(f"Error loading CSV/Models: {e}")

    try: sentiment_pipeline = pipeline("sentiment-analysis", model="5CD-AI/Vietnamese-Sentiment-visobert")
    except: pass
    try:
        semantic_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        semantic_model = AutoModel.from_pretrained("vinai/phobert-base-v2")
        candidate_vectors = get_semantic_vector(candidate_tags, semantic_tokenizer, semantic_model)
        candidate_vectors = candidate_vectors.reshape(len(candidate_tags), -1)
    except: pass

@app.post("/recommend", response_model=RecommendResponse)
async def handle_recommendation(request_data: RecommendRequest):
    if df is None: raise HTTPException(status_code=503, detail="Models not loaded")
    
    query_lower = request_data.query.lower()
    user_gps = request_data.user_gps
    
    print(f"\n[Request] '{query_lower}'")

    # --- BƯỚC 0: TÍNH KHOẢNG CÁCH TRƯỚC ---
    # Tính khoảng cách cho TOÀN BỘ quán so với user_gps
    # Để dùng cho việc lọc bán kính (Radius Filter)
    current_df = df.copy()
    distances = []
    for _, row in current_df.iterrows():
        d = 999.9
        if user_gps and len(user_gps) == 2:
            d = calculate_distance(user_gps[0], user_gps[1], row['lat'], row['lon'])
        distances.append(d)
    current_df['distance_km'] = distances

    # --- PIPELINE XỬ LÝ QUERY ---
    processed_query = " " + query_lower + " "
    sort_by = "taste"
    location_found = None
    
    # Lớp 1: Location Command
    for trig in LOCATION_TRIGGERS:
        for k, v in LOCATION_NAMES.items():
            if f" {trig} {k} " in processed_query:
                location_found = v
                processed_query = processed_query.replace(f" {trig} {k} ", " ")
                break
        if location_found: break
        
    # Lớp 1A & 1B
    for k, v in TAG_KNOWLEDGE_BASE.items():
        if f" {k} " in processed_query: processed_query = processed_query.replace(f" {k} ", f" {v} ")
    for k, v in SORT_KNOWLEDGE_BASE.items():
        if f" {k} " in processed_query: 
            sort_by = v
            processed_query = processed_query.replace(f" {k} ", " ")

    # Lớp 2: Fast Path
    extracted_tags = []
    sorted_candidates = sorted(candidate_tags, key=len, reverse=True)
    temp_query = processed_query
    for tag in sorted_candidates:
        if f" {tag.lower()} " in temp_query: 
            extracted_tags.append(tag)
            temp_query = temp_query.replace(f" {tag.lower()} ", " ")

    # Lớp 3: Semantic
    if not extracted_tags and semantic_model:
        try:
            qv = get_semantic_vector(query_lower, semantic_tokenizer, semantic_model)
            scores = cosine_similarity(qv, candidate_vectors).flatten()
            for i, s in enumerate(scores):
                if s > 0.6: extracted_tags.append(candidate_tags[i])
        except: pass
    
    final_query = " ".join(extracted_tags) if extracted_tags else query_lower
    
    # --- LỌC DỮ LIỆU (QUAN TRỌNG) ---
    
    if location_found:
        # Kịch bản 1: User nói "ở Hà Nội"
        # -> Lọc theo tên Địa danh, BỎ QUA GPS và Bán kính
        print(f"-> Lọc theo Location: {location_found}")
        filtered_df = current_df[current_df['district'].isin(location_found)]
        
    elif request_data.city_filter:
         # Kịch bản 2: Client test (cũ) gửi city_filter
         filtered_df = current_df[current_df['district'].str.contains(request_data.city_filter, case=False, na=False)]

    else:
        # Kịch bản 3: MẶC ĐỊNH (User không nói địa danh)
        # -> TỰ ĐỘNG LỌC BÁN KÍNH 20KM (Radius Filter)
        # Đây là bước sửa lỗi "Phở Hà Nội hiện ở HCM"
        print("-> Lọc theo Bán kính GPS (20km)")
        filtered_df = current_df[current_df['distance_km'] <= 20.0]
        
        # Nếu Client có gửi candidate_ids (kiểu cũ), lọc thêm
        if request_data.candidate_ids:
            filtered_df = filtered_df[filtered_df['id'].isin(request_data.candidate_ids)]

    if filtered_df.empty: return {"sort_by": sort_by, "scores": []}

    # --- LỌC CỨNG TAGS (Sửa lỗi Cơm Tấm/Cơm Gà) ---
    if extracted_tags:
        tags_pattern = '|'.join([re.escape(t) for t in extracted_tags])
        strict_filtered_df = filtered_df[filtered_df['tags'].str.contains(tags_pattern, case=False, na=False)]
        if not strict_filtered_df.empty:
            filtered_df = strict_filtered_df

    # --- TÍNH ĐIỂM TASTE ---
    # Chỉ tính điểm cho các quán đã lọc
    qv_3a = tfidf_vectorizer.transform([final_query])
    tm_filtered = tfidf_vectorizer.transform(filtered_df['tags'])
    taste_scores = cosine_similarity(qv_3a, tm_filtered).flatten()
    
    # Gán điểm Taste vào DF (để sort)
    # Lưu ý: filtered_df có thể là view, nên dùng .loc hoặc copy
    filtered_df = filtered_df.copy()
    filtered_df['S_taste'] = taste_scores

    # Lọc điểm > 0
    if final_query.strip():
         final_candidates = filtered_df[filtered_df['S_taste'] > 0.001]
         if final_candidates.empty and sort_by == 'distance':
             final_candidates = filtered_df
    else:
         final_candidates = filtered_df

    # --- SẮP XẾP ---
    if sort_by == 'distance' and user_gps:
        final_candidates = final_candidates.sort_values('distance_km', ascending=True)
    elif sort_by == 'price':
        final_candidates = final_candidates.sort_values('price', ascending=True)
    elif sort_by == 'rating':
        final_candidates = final_candidates.sort_values('rating', ascending=False)
    else:
        final_candidates = final_candidates.sort_values('S_taste', ascending=False)

    scores_list = []
    for _, row in final_candidates.iterrows():
        scores_list.append({
            "id": int(row['id']),
            "name": str(row['name']),
            "tags": str(row['tags']),
            "S_taste": float(row['S_taste']),
            "distance_km": float(row['distance_km']),
            "price": int(row['price'])
        })
        
    return {"sort_by": sort_by, "scores": scores_list}

@app.post("/sentiment", response_model=SentimentResponse)
async def handle_sentiment(request_data: SentimentRequest):
    if not sentiment_pipeline: raise HTTPException(503, "Model not loaded")
    return sentiment_pipeline(request_data.review)[0]

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=5000, reload=True)