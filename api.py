# ai-service/api.py
import pandas as pd
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import math
import re

# --- CẤU HÌNH ---
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "vietnomnom"      
COLLECTION_NAME = "restaurants"

# --- MODELS ---
class RecommendRequest(BaseModel):
    query: str 
    candidate_ids: Optional[List[str]] = [] 
    user_gps: Optional[List[float]] = None
    city_filter: Optional[str] = None

class TasteScore(BaseModel):
    id: str             
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
    # Kiểm tra dữ liệu đầu vào chặt chẽ hơn
    if lat1 is None or lat2 is None or lon1 is None or lon2 is None: return 999.0
    if pd.isna(lat1) or pd.isna(lat2) or pd.isna(lon1) or pd.isna(lon2): return 999.0
    
    try:
        # Ép kiểu float lần nữa để chắc chắn
        lat1, lon1 = float(lat1), float(lon1)
        lat2, lon2 = float(lat2), float(lon2)
        
        R = 6371 
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        a = math.sin(dLat/2) * math.sin(dLat/2) + \
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
            math.sin(dLon / 2) * math.sin(dLon / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    except Exception as e:
        return 999.0

def get_semantic_vector(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad(): outputs = model(**inputs)
    return outputs.last_hidden_state[:,0,:].numpy()

print("--- KHỞI ĐỘNG SERVER AI (V-FINAL-CLEAN-GPS) ---")
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
    'quận 1': 'Quận 1', 'q1': 'Quận 1', 'q 1': 'Quận 1',
    'quận 2': 'Quận 2', 'q2': 'Quận 2', 'q 2': 'Quận 2',
    'quận 3': 'Quận 3', 'q3': 'Quận 3', 'q 3': 'Quận 3',
    'quận 4': 'Quận 4', 'q4': 'Quận 4', 'q 4': 'Quận 4',
    'quận 5': 'Quận 5', 'q5': 'Quận 5', 'q 5': 'Quận 5',
    'quận 6': 'Quận 6', 'q6': 'Quận 6', 'q 6': 'Quận 6',
    'quận 7': 'Quận 7', 'q7': 'Quận 7', 'q 7': 'Quận 7',
    'quận 8': 'Quận 8', 'q8': 'Quận 8', 'q 8': 'Quận 8',
    'quận 9': 'Quận 9', 'q9': 'Quận 9', 'q 9': 'Quận 9',
    'quận 10': 'Quận 10', 'q10': 'Quận 10', 'q 10': 'Quận 10',
    'quận 11': 'Quận 11', 'q11': 'Quận 11', 'q 11': 'Quận 11',
    'quận 12': 'Quận 12', 'q12': 'Quận 12', 'q 12': 'Quận 12',
    'bình thạnh': 'Bình Thạnh', 'phú nhuận': 'Phú Nhuận',
    'gò vấp': 'Gò Vấp', 'tân bình': 'Tân Bình',
    'tân phú': 'Tân Phú', 'bình tân': 'Bình Tân',
    'thủ đức': 'Thủ Đức', 'nhà bè': 'Nhà Bè', 'bình chánh': 'Bình Chánh',
    'hóc môn': 'Hóc Môn', 'củ chi': 'Củ Chi', 'cần giờ': 'Cần Giờ',
    'sài gòn': 'TPHCM', 'hcm': 'TPHCM', 'tphcm': 'TPHCM'
}
LOCATION_TRIGGERS = ['ở', 'tại', 'đến', 'thuộc']
candidate_tags = [
    'bánh tằm', 'cháo lòng', 'bún quậy', 'nui xào', 'mì xào', 
    'bánh canh', 'bánh bột lọc', 'bánh bèo', 'bánh khọt', 'bánh hỏi',
    'cháo', 'nui', 'miến', 'heo quay', 'vịt quay', 'bò né',
    'lẩu gà lá é', 'gà nướng', 'cơm gà', 'bún chả cá', 'bún mắm',

    'bún bò', 'phở', 'cơm tấm', 'pizza', 'gà rán', 'bánh xèo', 'mì quảng', 'bún đậu', 'ốc', 'hải sản', 'sushi', 'lẩu', 'bò', 'hủ tiếu', 'nướng', 'bánh mì', 'cơm việt', 'dê', 'phá lấu', 'steak', 'ramen', 'dimsum', 'bánh canh', 'cua', 'mì ý', 'bia thủ công', 'chả cá', 'bingsu', 'xôi', 'cuốn', 'bò né', 'kem', 'xiên que', 'cơm niêu', 'cà phê vợt', 'cơm lam', 'đậu hũ', 'cay', 'ngọt', 'chay', 'mắm tôm', 'phô mai', 'miền trung', 'bắc', 'hàn quốc', 'ý', 'nhật', 'trung hoa', 'âu', 'miền nam', 'miền tây', 'vỉa hè', 'sang trọng', 'yên tĩnh', 'hẹn hò', 'truyền thống', 'nhậu', 'đêm', 'nhanh', 'buffet', 'mang đi', 'làm việc', 'gia đình', 'bình dân', 'dịch vụ', 'sáng', 'trưa', 'chè', 'ăn vặt', 'trà sữa', 'bánh', 'cơm', 'mì', 'gà', 'cá'
]

# [MỚI] Hàm trích xuất quận từ địa chỉ
def extract_district_from_address(address):
    if not isinstance(address, str): return ''
    addr_lower = address.lower()
    
    # Ưu tiên tìm các quận tên dài hoặc đặc biệt trước
    if 'thủ đức' in addr_lower: return 'Thủ Đức'
    if 'bình thạnh' in addr_lower: return 'Bình Thạnh'
    if 'phú nhuận' in addr_lower: return 'Phú Nhuận'
    if 'tân bình' in addr_lower: return 'Tân Bình'
    if 'gò vấp' in addr_lower: return 'Gò Vấp'
    if 'tân phú' in addr_lower: return 'Tân Phú'
    if 'bình tân' in addr_lower: return 'Bình Tân'
    if 'nhà bè' in addr_lower: return 'Nhà Bè'
    if 'bình chánh' in addr_lower: return 'Bình Chánh'
    if 'hóc môn' in addr_lower: return 'Hóc Môn'
    if 'củ chi' in addr_lower: return 'Củ Chi'
    if 'cần giờ' in addr_lower: return 'Cần Giờ'
    
    # Tìm quận số (Quận 1, Quận 10...)
    # Dùng regex để tránh nhầm lẫn (ví dụ "Quận 1" vs "Quận 12")
    for i in range(12, 0, -1):
        q_name = f"quận {i}"
        if q_name in addr_lower:
            return f"Quận {i}"
            
    return ''

# --- HÀM LÀM SẠCH DỮ LIỆU (QUAN TRỌNG) ---
def clean_coordinate(val):
    """Chuyển đổi tọa độ sang float, xử lý dấu phẩy"""
    try:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            # Thay dấu phẩy thành chấm và parse
            return float(val.replace(',', '.'))
        return 0.0
    except:
        return 0.0

@app.on_event("startup")
async def startup_event():
    global df, tfidf_vectorizer, tag_matrix, sentiment_pipeline, semantic_tokenizer, semantic_model, candidate_vectors
    
    try:
        print(f"Đang kết nối MongoDB: {DB_NAME}...")
        client = MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        
        cursor = collection.find({})
        data_list = []
        for doc in cursor:
            doc['id'] = str(doc['_id'])
            del doc['_id'] 
            data_list.append(doc)
        
        if not data_list:
            print("CẢNH BÁO: Database rỗng!")
            df = pd.DataFrame(columns=['id', 'tenQuan', 'tags', 'diemTrungBinh', 'giaCa', 'lat', 'lon'])
        else:
            df = pd.DataFrame(data_list)
            
            rename_map = {
                'tenQuan': 'name',
                'diemTrungBinh': 'rating',
                'giaCa': 'price',
                'diaChi': 'address'
            }
            df.rename(columns=rename_map, inplace=True)

            # 1. Xử lý Tags
            df['tags'] = df['tags'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
            df['tags'] = df['tags'].fillna('')
            
            # 2. Xử lý Tọa độ (FIX LỖI GPS)
            if 'lat' in df.columns:
                df['lat'] = df['lat'].apply(clean_coordinate)
            else:
                df['lat'] = 0.0

            if 'lon' in df.columns:
                df['lon'] = df['lon'].apply(clean_coordinate)
            else:
                df['lon'] = 0.0
            
            # Kiểm tra District
            print("Đang trích xuất dữ liệu Quận/Huyện...")
            df['district'] = df['address'].apply(extract_district_from_address)
            
            tfidf_vectorizer = TfidfVectorizer()
            tag_matrix = tfidf_vectorizer.fit_transform(df['tags'])
            print(f"AI Sẵn sàng! (Data từ Mongo: {len(df)} quán)")

    except Exception as e:
        print(f"Lỗi load MongoDB: {e}")
        df = None

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
    if df is None or df.empty: 
        raise HTTPException(status_code=503, detail="Database empty")
    
    query_lower = request_data.query.lower()
    user_gps = request_data.user_gps
    print(f"\n[Request] '{query_lower}'")

    current_df = df.copy()
    distances = []
    
    # Tính khoảng cách
    for _, row in current_df.iterrows():
        d = 999.9
        if user_gps and len(user_gps) == 2:
            d = calculate_distance(user_gps[0], user_gps[1], row['lat'], row['lon'])
        distances.append(d)
    current_df['distance_km'] = distances

    # --- LOGIC XỬ LÝ QUERY (GIỮ NGUYÊN) ---
    processed_query = " " + query_lower + " "
    sort_by = "taste"
    location_found = None
    locations_to_filter = []

    sorted_loc_keys = sorted(LOCATION_NAMES.keys(), key=len, reverse=True)
    
    for k in sorted_loc_keys:
        # Tìm chính xác từ khóa (có khoảng trắng bao quanh để tránh matching một phần)
        if f" {k} " in processed_query:
            real_name = LOCATION_NAMES[k]
            locations_to_filter.append(real_name)
            # Xóa từ khóa khỏi query để không ảnh hưởng tới tìm món ăn
            processed_query = processed_query.replace(f" {k} ", " ")
            location_found = real_name # Lưu lại để dùng cho logic cũ nếu cần

    locations_to_filter = list(set(locations_to_filter))
        
    for k, v in TAG_KNOWLEDGE_BASE.items():
        if f" {k} " in processed_query: processed_query = processed_query.replace(f" {k} ", f" {v} ")
    for k, v in SORT_KNOWLEDGE_BASE.items():
        if f" {k} " in processed_query: 
            sort_by = v
            processed_query = processed_query.replace(f" {k} ", " ")

    extracted_tags = []
    sorted_candidates = sorted(candidate_tags, key=len, reverse=True)
    temp_query = processed_query
    for tag in sorted_candidates:
        if f" {tag.lower()} " in temp_query: 
            extracted_tags.append(tag)
            temp_query = temp_query.replace(f" {tag.lower()} ", " ")

    if not extracted_tags and semantic_model:
        try:
            qv = get_semantic_vector(query_lower, semantic_tokenizer, semantic_model)
            scores = cosine_similarity(qv, candidate_vectors).flatten()
            for i, s in enumerate(scores):
                if s > 0.6: extracted_tags.append(candidate_tags[i])
        except: pass
    
    final_query = " ".join(extracted_tags) if extracted_tags else query_lower
    
    # --- LỌC DỮ LIỆU ---
    if locations_to_filter:
        print(f"-> Lọc theo Quận: {locations_to_filter}")
        # Lọc các quán có district nằm trong danh sách tìm thấy
        filtered_df = current_df[current_df['district'].isin(locations_to_filter)]
    elif request_data.city_filter:
         filtered_df = current_df[current_df['district'].str.contains(request_data.city_filter, case=False, na=False)]
    else:
        # Logic Lọc Bán kính an toàn
        if user_gps and len(user_gps) == 2:
            print("-> Lọc theo Bán kính GPS (20km)")
            filtered_df = current_df[current_df['distance_km'] <= 20.0]
        else:
            print("-> Không có GPS: Tìm toàn bộ quán.")
            filtered_df = current_df

        if request_data.candidate_ids:
            filtered_df = filtered_df[filtered_df['id'].isin(request_data.candidate_ids)]

    if filtered_df.empty: return {"sort_by": sort_by, "scores": []}

    if extracted_tags:
        tags_pattern = '|'.join([re.escape(t) for t in extracted_tags])
        strict_filtered_df = filtered_df[filtered_df['tags'].str.contains(tags_pattern, case=False, na=False)]
        if not strict_filtered_df.empty: filtered_df = strict_filtered_df

    qv_3a = tfidf_vectorizer.transform([final_query])
    tm_filtered = tfidf_vectorizer.transform(filtered_df['tags'])
    taste_scores = cosine_similarity(qv_3a, tm_filtered).flatten()
    
    filtered_df = filtered_df.copy()
    filtered_df['S_taste'] = taste_scores

    if final_query.strip():
         final_candidates = filtered_df[filtered_df['S_taste'] > 0.001]
         if final_candidates.empty and sort_by == 'distance': final_candidates = filtered_df
    else: final_candidates = filtered_df

    # Sắp xếp
    if sort_by == 'distance' and user_gps:
        # [QUAN TRỌNG] Sắp xếp lại theo khoảng cách chính xác
        final_candidates = final_candidates.sort_values('distance_km', ascending=True)
    elif sort_by == 'price':
        final_candidates = final_candidates.sort_values('price', ascending=True)
    elif sort_by == 'rating':
        final_candidates = final_candidates.sort_values('rating', ascending=False)
    else:
        final_candidates = final_candidates.sort_values('S_taste', ascending=False)

    scores_list = []
    for _, row in final_candidates.iterrows():
        p_val = 0
        try: p_val = int(str(row['price']).replace('.', '').replace(',', ''))
        except: pass

        scores_list.append({
            "id": str(row['id']),
            "name": str(row['name']),
            "tags": str(row['tags']),
            "S_taste": float(row['S_taste']),
            "distance_km": float(row['distance_km']),
            "price": p_val
        })
        
    return {"sort_by": sort_by, "scores": scores_list}

@app.post("/sentiment", response_model=SentimentResponse)
async def handle_sentiment(request_data: SentimentRequest):
    if not sentiment_pipeline: raise HTTPException(503, "Model not loaded")
    return sentiment_pipeline(request_data.review)[0]

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=5000, reload=True)