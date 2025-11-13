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
print("--- KHỞI ĐỘNG SERVER AI (100% TỰ HOST - V-FINAL-V4.1) ---")
app = FastAPI(title="Smart Tourism AI Toolbox API (Self-Hosted)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- (Biến Global cho Model - Không thay đổi) ---
df = None
tfidf_vectorizer = None
tag_matrix = None
sentiment_pipeline = None
semantic_tokenizer = None
semantic_model = None
candidate_vectors = None

# --- (Tải "Lớp Tri Thức" (ĐÃ SỬA) và "Tags Sạch") ---

# LỚP 1A: (Dùng để "dịch" Tag Mơ hồ -> Tag Sạch)
TAG_KNOWLEDGE_BASE = {
    'lãng mạn': 'hẹn hò', 'sang chảnh': 'sang trọng', 'đắt tiền': 'sang trọng',
    'thoải mái': 'yên tĩnh', 'nhanh gọn': 'nhanh', 'tụ tập': 'nhậu bạn bè', 'bạn bè': 'nhậu bạn bè',
    'đồng quê': 'cơm việt truyền thống',
    'đặc sản huế': 'bún bò miền trung', 
    'đặc sản hà nội': 'phở bún chả bắc'
}
# LỚP 1B: (Dùng để suy luận 'sort_by')
SORT_KNOWLEDGE_BASE = {
    'gần': 'distance', 'rẻ': 'price', 'ngon': 'rating',
    'tốt nhất': 'rating', 'đánh giá cao': 'rating'
}

# --- (SỬA LỖI: Logic V4 - Giả sử CSV đã "sạch") ---
LOCATION_TRIGGERS = ['ở', 'tại', 'đến', 'thuộc']
LOCATION_NAMES = {
    'hà nội': 'Hà Nội',
    'sài gòn': 'TPHCM',
    'tp.hcm': 'TPHCM',
    'hồ chí minh': 'TPHCM'
}
# --- (HẾT SỬA LỖI 1C) ---

# LỚP 2: DANH SÁCH TAGS "SẠCH" (150 nhà hàng)
candidate_tags = [
    'bún bò', 'phở', 'cơm tấm', 'pizza', 'gà rán', 'bánh xèo', 'mì quảng', 'bún đậu', 'ốc', 'hải sản', 'sushi', 'lẩu', 'bò 7 món', 'hủ tiếu', 'đồ nướng', 'bánh mì', 'cơm việt', 'lẩu dê', 'phá lấu', 'steak', 'ramen', 'dimsum', 'lẩu trung hoa', 'bánh canh cua', 'mì ý', 'bia thủ công', 'chả cá', 'cuốn', 'bò né', 'cơm niêu', 'cơm lam', 'cơm gà', 'bánh cuốn', 'lẩu nấm', 'cháo sườn', 'mì vằn thắn', 'vịt quay', 'bánh bột lọc', 'bò kho', 'bún mắm', 'bánh tráng nướng', 'nem nướng', 'lẩu cá kèo', 'bún riêu', 'mì vịt tiềm', 'hủ tiếu mực', 'gà nướng', 'cơm văn phòng', 'chè thái', 'ăn vặt', 'tráng miệng', 'trà sữa', 'bingsu', 'kem', 'xiên que', 'đậu hũ', 'tàu hủ', 'bánh ngọt', 'xôi gà', 'xôi bắp', 'king roti', 'cay', 'ngọt', 'chay', 'mắm tôm', 'phô mai', 'xối mỡ', '7 cấp độ', 'chua', 'miền trung', 'bắc', 'hàn quốc', 'ý', 'nhật', 'trung hoa', 'âu', 'miền nam', 'miền tây', 'nam vang', 'á', 'vỉa hè', 'sang trọng', 'yên tĩnh', 'hẹn hò', 'truyền thống', 'nhậu', 'đêm', 'nhanh', 'buffet', 'mang đi', 'làm việc', 'gia đình', 'bạn bè', 'bình dân', 'dịch vụ', 'sáng', 'trưa', 'băng chuyền', 'điểm tâm', 'chuỗi', 'cà phê vợt',
    'tái lăn', 'bánh tôm', 'miến lươn', 'nộm bò khô', 'bún thang', 'cháo trai', 'lẩu riêu cua', 'nem chua nướng', 'bánh gối', 'bò nhúng dấm', 'vịt', 'bún ốc', 'bánh đa cua', 'bánh giò', 'bánh mì chảo', 'bò bít tết', 'bún cá', 'gà tần', 'chân gà nướng', 'bún ngan', 'quẩy', 'trà chanh', 'phở xào', 'bánh khúc', 'lẩu thái', 'mì gà tần', 'bún bò nam bộ', 'gà không lối thoát', 'bánh đúc nóng', 'nem lụi', 'bò nầm nướng', 'bún dọc mùng', 'lẩu đài loan', 'cơm rang', 'bánh đa trộn', 'cơm sườn', 'kem xôi', 'bánh tráng trộn', 'chè sầu', 'nem rán', 'phở cuốn', 'sinh viên'
]

# --- (Hàm get_semantic_vector - Không thay đổi) ---
def get_semantic_vector(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:,0,:].numpy()

# --- SỰ KIỆN STARTUP (Không thay đổi) ---
@app.on_event("startup")
async def startup_event():
    global df, tfidf_vectorizer, tag_matrix, sentiment_pipeline
    global semantic_tokenizer, semantic_model, candidate_vectors

    # === Tải AI 1: GĐ 3A (Scikit-learn) ===
    print("Đang tải AI GĐ 3A (Scikit-learn - So khớp)...")
    try:
        BASE_DIR = Path(__file__).resolve().parent
        CSV_PATH = os.path.normpath(BASE_DIR / 'restaurants.csv') 
        print(f"Đang đọc file CSV từ: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
        
        # (Dọn dẹp ID - Ép về INT)
        df['id'] = pd.to_numeric(df['id'], errors='coerce') 
        df = df.dropna(subset=['id'])
        df['id'] = df['id'].astype(int) 
        df['tags'] = df['tags'].fillna('')
        df['district'] = df['district'].fillna('').astype(str)
        
        tfidf_vectorizer = TfidfVectorizer()
        tag_matrix = tfidf_vectorizer.fit_transform(df['tags'])
        print(f"AI GĐ 3A (Scikit-learn) đã sẵn sàng! Đã học {tag_matrix.shape[1]} từ vựng.")
    except Exception as e:
        print(f"LỖI GĐ 3A: {e}")

    # === Tải AI 2 (Sentiment) ===
    print("Đang tải AI GĐ 3B (5CD-AI Sentiment)...")
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model="5CD-AI/Vietnamese-Sentiment-visobert")
        print("AI GĐ 3B (5CD-AI Sentiment) đã sẵn sàng!")
    except Exception as e:
        print(f"LỖI GĐ 3B (Sentiment): {e}")

    # === Tải AI 3 (PhoBERT) ===
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
        
    # 1. Nhận input
    query_lower = request_data.query.lower()
    candidate_ids_int = request_data.candidate_ids
    print(f"\n[Request /recommend] Nhận query: '{query_lower}'")
    
    # 2. CHẠY "PIPELINE 3 LỚP" (Xử lý Query)
    
    processed_query = " " + query_lower + " "
    sort_by = "taste" 
    location_command_found = None 

    # --- LỚP 1C (MỚI): Phát hiện LỆNH LOCATION (Ưu tiên cao nhất) ---
    for trigger in LOCATION_TRIGGERS:
        for loc_key, loc_name in LOCATION_NAMES.items():
            if f" {trigger} {loc_key} " in processed_query:
                location_command_found = loc_name 
                processed_query = processed_query.replace(f" {trigger} {loc_key} ", " ")
                print(f"[Lớp 1C] Phát hiện Lệnh Location: '{location_command_found}'")
                break
        if location_command_found: break

    # --- LỚP 1A & 1B (Tri Thức: Dịch Tag & Sort) ---
    for keyword, replacement in TAG_KNOWLEDGE_BASE.items():
        if f" {keyword} " in processed_query: 
            processed_query = processed_query.replace(f" {keyword} ", f" {replacement} ")
    for keyword, replacement in SORT_KNOWLEDGE_BASE.items():
        if f" {keyword} " in processed_query: 
            sort_by = replacement
            processed_query = processed_query.replace(f" {keyword} ", " ")
            
    print(f"[Lớp 1] Query đã qua Lớp Tri Thức: '{processed_query}', Sort_by: '{sort_by}'")
    
    # =================================================================
    # === BẮT ĐẦU SỬA LỖI (LOGIC FALLBACK) ===
    # =================================================================
    
    # --- LỚP 2: LỐI ĐI NHANH (Fast Path) ---
    extracted_tags_set = set()
    for tag in candidate_tags:
        if f" {tag.lower()} " in processed_query: 
            extracted_tags_set.add(tag)
            
    # --- LOGIC "FALLBACK" ("Cầu dao an toàn") ---
    
    # NẾU Lớp 2 (Fast Path) tìm thấy tag (Set không rỗng), 
    # DỪNG LẠI và DÙNG NGAY.
    if extracted_tags_set:
        print(f"[Lớp 2] THÀNH CÔNG. Dùng {len(extracted_tags_set)} tag (Fast Path). Bỏ qua Lớp 3.")
        extracted_tags = list(extracted_tags_set)
    
    # CHỈ CHẠY Lớp 3 (Semantic) NẾU Lớp 2 thất bại (Set rỗng)
    else:
        print("[Lớp 2] Thất bại. Chuyển sang Lớp 3 (Semantic Fallback)...")
        extracted_tags = [] # Khởi tạo list rỗng
        
        if semantic_model is not None:
            threshold = 0.6 # Ngưỡng semantic
            query_to_search = processed_query.strip()
            
            if query_to_search: # Chỉ chạy nếu query không rỗng
                try:
                    query_vector = get_semantic_vector(query_to_search, semantic_tokenizer, semantic_model)
                    semantic_scores = cosine_similarity(query_vector, candidate_vectors).flatten()
                    
                    for i, score in enumerate(semantic_scores):
                        if score > threshold:
                            extracted_tags.append(candidate_tags[i]) # Thêm vào list
                    print(f"[Lớp 3] Tìm thấy {len(extracted_tags)} tag (Semantic).")
                except Exception as e:
                    print(f"LỖI LỚP 3 (Semantic): {e}")
            else:
                print("[Lớp 3] Bỏ qua (processed_query rỗng).")
        else:
            print("Lớp 3 (PhoBERT) chưa tải, bỏ qua.")
    
    # =================================================================
    # === KẾT THÚC SỬA LỖI (LOGIC FALLBACK) ===
    # =================================================================
    
    final_query = " ".join(extracted_tags)
    if not final_query.strip(): 
        print("[Pipeline] Query cuối cùng rỗng. Trả về rỗng.")
        return {"sort_by": sort_by, "scores": []} 
    print(f"[Pipeline] Query đã xử lý (Cuối cùng): '{final_query}'")

    # 3. CHẠY GĐ 3A (LỌC & TÍNH ĐIỂM "TASTE")
    
    # --- (LỌC LOCATION THEO LOGIC CỦA BẠN) ---
    if location_command_found:
        print(f"Lọc theo LOCATION: '{location_command_found}'. Bỏ qua GPS.")
        filtered_df = df[df['district'].str.lower() == location_command_found.lower()]
    else:
        print(f"Lọc theo GPS (candidate_ids).")
        filtered_df = df[df['id'].isin(candidate_ids_int)]
    
    if filtered_df.empty:
        print("DEBUG: filtered_df rỗng (Lọc Location/GPS thất bại).")
        return {"sort_by": sort_by, "scores": []} 

    # --- (LỌC LỚP 2 - TAG & Tính "Taste") ---
    query_vector_3a = tfidf_vectorizer.transform([final_query])
    filtered_tag_matrix = tfidf_vectorizer.transform(filtered_df['tags'])
    cosine_scores_3a = cosine_similarity(query_vector_3a, filtered_tag_matrix).flatten()
    
    filtered_df = filtered_df.copy()
    filtered_df['S_taste'] = cosine_scores_3a
    
    # (In debug scores TRƯỚC KHI LỌC)
    # print("\n--- DEBUG: TOP 10 SCORES (TRƯỚC KHI LỌC) ---")
    # print(filtered_df[['id', 'name', 'S_taste']].sort_values('S_taste', ascending=False).head(10))
    # print("-------------------------------------------\n")

    # =================================================================
    # === SỬA LỖI 2: HẠ NGƯỠNG S_TASTE ===
    # =================================================================
    # (Hạ ngưỡng xuống 0.1 để bắt các kết quả TF-IDF thấp nhưng liên quan)
    final_candidates_df = filtered_df[filtered_df['S_taste'] > 0.1] 
    
    # --- (Phần trả về 'name', 'tags' giữ nguyên) ---
    scores_list = []
    for index, restaurant_data in final_candidates_df.iterrows():
        scores_list.append({
            "id": int(restaurant_data['id']),
            "name": str(restaurant_data['name']),
            "tags": str(restaurant_data['tags']),
            "S_taste": float(restaurant_data['S_taste'])
        })
        
    print(f"Trả về {len(scores_list)} điểm 'Taste' (với S_taste > 0.1) và 'sort_by: {sort_by}'.")
    return {"sort_by": sort_by, "scores": scores_list}


# --- Endpoint 2: Phân tích Review (Sentiment) ---
@app.post("/sentiment", response_model=SentimentResponse)
async def handle_sentiment(request_data: SentimentRequest):
    if sentiment_pipeline is None:
        raise HTTPException(status_code=503, detail="Sentiment model not loaded")
    review_text = request_data.review
    try:
        result = sentiment_pipeline(review_text)
        return result[0] 
    except Exception as e:
        print(f"LỖI GĐ 3B (Sentiment): {e}")
        raise HTTPException(status_code=500, detail="AI model failed")

# -----------------------------------------------------------------
# BƯỚC 4: CHẠY SERVER (Bằng lệnh Terminal)
# -----------------------------------------------------------------
# (Hãy chạy bằng lệnh: py -m uvicorn api:app --reload --port 5000)