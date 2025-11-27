# ai-service/api.py
import pandas as pd
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import math
import os

# --- CẤU HÌNH ---
MONGO_URI = "mongodb+srv://quockhanh:quockhanh1234@vietnomnom.sxnsf4y.mongodb.net/?retryWrites=true&w=majority&appName=VietNomNom" 
DB_NAME = "VietNomNom"      
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

# --- LOGIC APP ---
print("--- KHỞI ĐỘNG SERVER AI (FINAL PRODUCTION) ---")
app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# --- GLOBAL VARIABLES ---
df = None
tfidf_vectorizer = None
sentiment_pipeline = None

# --- CONSTANTS ---
TAG_KNOWLEDGE_BASE = {
    'lãng mạn': 'hẹn hò', 'sang chảnh': 'sang trọng', 'đắt tiền': 'sang trọng',
    'thoải mái': 'yên tĩnh', 'nhanh gọn': 'nhanh', 'tụ tập': 'nhậu bạn bè', 
    'đồng quê': 'cơm việt truyền thống', 'đặc sản huế': 'bún bò miền trung', 
    'đặc sản hà nội': 'phở bún chả bắc', 'đặc sản sài gòn': 'cơm tấm miền nam'
}

EN_VI_MAPPING = {
    'beef noodle': 'bún bò', 'pho': 'phở', 'broken rice': 'cơm tấm', 
    'rice': 'cơm', 'bread': 'bánh mì', 'hotpot': 'lẩu', 'steak': 'bít tết', 
    'seafood': 'hải sản', 'coffee': 'cà phê', 'vegetarian': 'chay',
    'spicy': 'cay', 'delicious': 'ngon', 'cheap': 'rẻ', 'near': 'gần',
    'district': 'quận', 'city': 'thành phố', 'restaurant': 'nhà hàng',
    'late night': 'ăn đêm', 'morning': 'sáng', 'noon': 'trưa',
    'best': 'ngon nhất', 'good': 'ngon'
}

LOCATION_NAMES = {
    'quận 1': 'Quận 1', 'q1': 'Quận 1', 'quận 2': 'Quận 2', 'q2': 'Quận 2',
    'quận 3': 'Quận 3', 'q3': 'Quận 3', 'quận 4': 'Quận 4', 'q4': 'Quận 4',
    'quận 5': 'Quận 5', 'q5': 'Quận 5', 'quận 6': 'Quận 6', 'q6': 'Quận 6',
    'quận 7': 'Quận 7', 'q7': 'Quận 7', 'quận 8': 'Quận 8', 'q8': 'Quận 8',
    'quận 9': 'Quận 9', 'q9': 'Quận 9', 'quận 10': 'Quận 10', 'q10': 'Quận 10',
    'quận 11': 'Quận 11', 'q11': 'Quận 11', 'quận 12': 'Quận 12', 'q12': 'Quận 12',
    'bình thạnh': 'Bình Thạnh', 'phú nhuận': 'Phú Nhuận', 'gò vấp': 'Gò Vấp',
    'tân bình': 'Tân Bình', 'tân phú': 'Tân Phú', 'bình tân': 'Bình Tân',
    'thủ đức': 'Thủ Đức', 'sài gòn': 'TPHCM', 'hcm': 'TPHCM'
}

candidate_tags = [
    'sáng', 'trưa', 'chiều', 'tối', 'đêm', 'ăn đêm', 
    'bún bò', 'phở', 'cơm tấm', 'pizza', 'gà rán', 'bánh xèo', 'mì quảng', 
    'bún đậu', 'ốc', 'hải sản', 'sushi', 'lẩu', 'bò', 'hủ tiếu', 'nướng', 
    'bánh mì', 'cơm việt', 'dê', 'phá lấu', 'steak', 'ramen', 'dimsum', 
    'cua', 'mì ý', 'bia thủ công', 'chả cá', 'bingsu', 'xôi', 'cuốn', 
    'kem', 'xiên que', 'cơm niêu', 'cà phê vợt', 'cơm lam', 'đậu hũ', 
    'cay', 'ngọt', 'chay', 'mắm tôm', 'phô mai', 'vỉa hè', 'sang trọng', 
    'yên tĩnh', 'hẹn hò', 'truyền thống', 'nhậu', 'nhanh', 'buffet', 
    'mang đi', 'gia đình', 'bình dân', 'chè', 'ăn vặt', 'trà sữa'
]

# Tags bắt buộc (Hard Filter) - Chỉ thời gian
PRIORITY_TAGS = ['đêm', 'sáng', 'trưa', 'chiều', 'tối', 'ăn đêm']

# Tags tính từ (Dùng để loại trừ khi xác định Món Chính để lọc Strict)
# Đã thêm 'ngon', 'tốt nhất' vào đây để tránh hệ thống tưởng đây là tên món ăn
ADJECTIVE_TAGS = ['rẻ', 'gần', 'ngon', 'tốt', 'nhanh', 'đẹp', 'vỉa hè', 'sang trọng', 'yên tĩnh', 'nổi tiếng', 'nhất']

# --- HELPER FUNCTIONS ---
def clean_coordinate(val):
    try:
        if isinstance(val, (int, float)): return float(val)
        if isinstance(val, str): return float(val.replace(',', '.'))
        return 0.0
    except: return 0.0

def extract_district_from_address(address):
    if not isinstance(address, str): return ''
    addr_lower = address.lower()
    for k, v in LOCATION_NAMES.items():
        if k in addr_lower and 'quận' in k: return v
    if 'thủ đức' in addr_lower: return 'Thủ Đức'
    if 'bình thạnh' in addr_lower: return 'Bình Thạnh'
    if 'phú nhuận' in addr_lower: return 'Phú Nhuận'
    if 'tân bình' in addr_lower: return 'Tân Bình'
    if 'gò vấp' in addr_lower: return 'Gò Vấp'
    if 'tân phú' in addr_lower: return 'Tân Phú'
    if 'bình tân' in addr_lower: return 'Bình Tân'
    return ''

def translate_query(query):
    query_lower = query.lower()
    for k, v in EN_VI_MAPPING.items():
        if f" {k} " in f" {query_lower} ":
            query_lower = query_lower.replace(k, v)
    return query_lower

def calculate_distance(lat1, lon1, lat2, lon2):
    if any(x is None for x in [lat1, lon1, lat2, lon2]): return 999.0
    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
        R = 6371 
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    except: return 999.0

# --- STARTUP ---
@app.on_event("startup")
async def startup_event():
    global df, tfidf_vectorizer, sentiment_pipeline
    print("-> Đang load dữ liệu...")
    try:
        client = MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        data_list = list(collection.find({}))
        
        if not data_list:
            df = pd.DataFrame(columns=['id', 'name', 'tags', 'rating', 'price', 'lat', 'lon', 'district'])
        else:
            temp_df = pd.DataFrame(data_list)
            temp_df['id'] = temp_df['_id'].astype(str)
            rename_map = {'tenQuan': 'name', 'diemTrungBinh': 'rating', 'giaCa': 'price', 'diaChi': 'address'}
            temp_df.rename(columns=rename_map, inplace=True)
            
            temp_df['tags'] = temp_df['tags'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x)).fillna('')
            if 'lat' in temp_df.columns: temp_df['lat'] = temp_df['lat'].apply(clean_coordinate)
            else: temp_df['lat'] = 0.0
            if 'lon' in temp_df.columns: temp_df['lon'] = temp_df['lon'].apply(clean_coordinate)
            else: temp_df['lon'] = 0.0
            temp_df['district'] = temp_df['address'].apply(extract_district_from_address)
            
            df = temp_df
            tfidf_vectorizer = TfidfVectorizer()
            tag_matrix = tfidf_vectorizer.fit_transform(df['tags'])
            print(f"-> Load thành công {len(df)} quán ăn.")
            
    except Exception as e:
        print(f"!!! Lỗi Critical khi load DB: {e}")
        df = pd.DataFrame()

    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model="5CD-AI/Vietnamese-Sentiment-visobert")
    except: pass

@app.get("/")
async def root():
    status = "Active" if df is not None and not df.empty else "Empty Data"
    return {"status": "AI Service Running", "data_status": status}

@app.post("/recommend", response_model=RecommendResponse)
async def handle_recommendation(request_data: RecommendRequest):
    if df is None or df.empty:
        raise HTTPException(status_code=503, detail="Database chưa sẵn sàng")
    
    # 1. PRE-PROCESS
    original_query = request_data.query
    processed_query = translate_query(original_query)
    user_gps = request_data.user_gps
    
    current_df = df.copy()
    
    # Tính khoảng cách
    if user_gps and len(user_gps) == 2:
        current_df['distance_km'] = current_df.apply(lambda row: calculate_distance(user_gps[0], user_gps[1], row['lat'], row['lon']), axis=1)
    else:
        current_df['distance_km'] = 999.0
        
    temp_query = " " + processed_query.lower() + " "
    
    # Mapping Synonyms
    for k, v in TAG_KNOWLEDGE_BASE.items():
        if f" {k} " in temp_query: temp_query = temp_query.replace(f" {k} ", f" {v} ")

    # Extract Location
    locations_to_filter = []
    for k, v in LOCATION_NAMES.items():
        if f" {k} " in temp_query:
            locations_to_filter.append(v)
            temp_query = temp_query.replace(f" {k} ", " ")
    
    # Extract Tags
    extracted_tags = []
    sorted_candidates = sorted(candidate_tags, key=len, reverse=True)
    for tag in sorted_candidates:
        if f" {tag} " in temp_query:
            extracted_tags.append(tag)
            temp_query = temp_query.replace(f" {tag} ", " ")
            
    # Phân loại Tags
    mandatory_tags = [t for t in extracted_tags if t in PRIORITY_TAGS]
    
    # --- BƯỚC 1: LỌC CƠ BẢN (Location, GPS) ---
    filtered_df = current_df
    
    if locations_to_filter:
        filtered_df = filtered_df[filtered_df['district'].isin(locations_to_filter)]
    elif request_data.city_filter:
        filtered_df = filtered_df[filtered_df['district'].str.contains(request_data.city_filter, case=False, na=False)]
    else:
        if user_gps and len(user_gps) == 2:
            filtered_df = filtered_df[filtered_df['distance_km'] <= 20.0]
            
    # --- BƯỚC 2: LỌC THỜI GIAN (Hard Filter) ---
    if mandatory_tags:
        for tag in mandatory_tags:
            filtered_df = filtered_df[filtered_df['tags'].str.contains(tag, case=False, na=False)]
            
    if filtered_df.empty:
        return {"sort_by": "relevance", "scores": []}
        
    # --- BƯỚC 3: LỌC MÓN ĂN (Strict Dish Filter) ---
    # Xác định món chính bằng cách loại bỏ các từ tính từ (rẻ, ngon, gần...)
    target_dish_tags = [t for t in extracted_tags if t not in ADJECTIVE_TAGS and t not in PRIORITY_TAGS]
    
    if target_dish_tags:
        matches = []
        for index, row in filtered_df.iterrows():
            row_tags = str(row['tags']).lower()
            # Quán phải chứa ít nhất 1 món trong query
            if any(dish in row_tags for dish in target_dish_tags):
                matches.append(index)
        
        if len(matches) > 0:
            filtered_df = filtered_df.loc[matches]

    # --- BƯỚC 4: CHẤM ĐIỂM (Scoring) ---
    final_query_text = " ".join(extracted_tags) if extracted_tags else processed_query
    
    try:
        qv = tfidf_vectorizer.transform([final_query_text])
        tm = tfidf_vectorizer.transform(filtered_df['tags'])
        base_scores = cosine_similarity(qv, tm).flatten()
    except:
        base_scores = [0.0] * len(filtered_df)

    # Dish Boosting (Cộng điểm nếu tên món xuất hiện trong Tên quán)
    boosted_scores = []
    for idx, row in enumerate(filtered_df.itertuples()):
        score = base_scores[idx]
        row_name = str(row.name).lower()
        if target_dish_tags:
            for dish in target_dish_tags:
                if dish in row_name:
                    score += 0.3 # Cộng điểm nếu tên quán có tên món
        boosted_scores.append(score)

    filtered_df = filtered_df.copy()
    filtered_df['S_taste'] = boosted_scores

    final_candidates = filtered_df

    # --- BƯỚC 5: SẮP XẾP (Sorting) ---
    sort_by = "relevance"
    query_lower = processed_query.lower()
    
    if "gần" in query_lower:
        sort_by = "distance"
        final_candidates = final_candidates.sort_values('distance_km', ascending=True)
    elif "rẻ" in query_lower:
        sort_by = "price"
        final_candidates = final_candidates.sort_values('price', ascending=True)
    # [MỚI] Sort theo Rating
    elif any(x in query_lower for x in ['ngon', 'tốt nhất', 'nổi tiếng', 'rating', 'đánh giá']):
        sort_by = "rating"
        final_candidates = final_candidates.sort_values('rating', ascending=False)
    else:
        final_candidates = final_candidates.sort_values('S_taste', ascending=False)
        
    scores_list = []
    for _, row in final_candidates.head(20).iterrows():
        scores_list.append({
            "id": str(row['id']),
            "name": str(row['name']),
            "tags": str(row['tags']),
            "S_taste": float(row.get('S_taste', 0.0)),
            "distance_km": float(row.get('distance_km', 999.0)),
            "price": int(row['price']) if pd.notna(row['price']) and str(row['price']).isdigit() else 0
        })
        
    return {"sort_by": sort_by, "scores": scores_list}

@app.post("/sentiment", response_model=SentimentResponse)
async def handle_sentiment(request_data: SentimentRequest):
    if not sentiment_pipeline: 
        return {"label": "neutral", "score": 0.5} 
    res = sentiment_pipeline(request_data.review)[0]
    return {"label": res['label'], "score": res['score']}

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=5000, reload=True)