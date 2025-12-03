# ai-service/api.py
import pandas as pd
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import math
import os
import io
from PIL import Image
from ultralytics import YOLO

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
print("--- KHỞI ĐỘNG SERVER AI (FINAL PRODUCTION - FIX HANOI SEARCH) ---")
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
yolo_model = None

# --- CONSTANTS ---
TAG_KNOWLEDGE_BASE = {
    # Về không gian/nhu cầu
    'lãng mạn': 'hẹn hò', 'sang chảnh': 'sang trọng', 'đắt tiền': 'sang trọng', 'luxury': 'sang trọng',
    'thoải mái': 'yên tĩnh', 'nhanh gọn': 'nhanh', 'tụ tập': 'nhậu', 'nhậu nhẹt': 'nhậu',
    'bình dân': 'rẻ', 'hạt dẻ': 'rẻ', 'sinh viên': 'rẻ',
    'mát mẻ': 'máy lạnh', 'điều hòa': 'máy lạnh',
    'view đẹp': 'đẹp', 'sống ảo': 'đẹp',
    
    # Về món ăn (Vùng miền)
    'đồng quê': 'cơm việt', 'cơm bắc': 'cơm việt', 'cơm niêu': 'cơm việt',
    'đặc sản huế': 'bún bò huế', 'món huế': 'bún bò huế',
    'đặc sản hà nội': 'bún chả', 'phở bắc': 'phở',
    'đặc sản sài gòn': 'cơm tấm', 
    'đồ nướng': 'nướng', 'bbq': 'nướng',
    'hải sản tươi sống': 'hải sản'
}

EN_VI_MAPPING = {
    # -- Món nước --
    'beef noodle': 'bún bò', 'beef noodle soup': 'bún bò', 'bun bo': 'bún bò',
    'pho': 'phở', 'noodle soup': 'phở', 'chicken noodle': 'phở gà',
    'crab noodle': 'bún riêu', 'snail noodle': 'bún ốc',
    'fish noodle': 'bún cá',
    'hu tieu': 'hủ tiếu', 'vermicelli': 'bún', 'glass noodle': 'miến',
    'ramen': 'mì nhật', 'udon': 'mì udon',
    
    # -- Cơm & Món mặn --
    'broken rice': 'cơm tấm', 'com tam': 'cơm tấm', 'pork chop rice': 'cơm sườn',
    'chicken rice': 'cơm gà', 'fried rice': 'cơm chiên',
    'rice': 'cơm', 'sticky rice': 'xôi',
    'braised pork': 'thịt kho', 'catfish': 'cá kho',
    
    # -- Bánh & Ăn vặt --
    'bread': 'bánh mì', 'baguette': 'bánh mì', 'sandwich': 'bánh mì',
    'pancake': 'bánh xèo', 'sizzling cake': 'bánh xèo',
    'spring roll': 'gỏi cuốn', 'summer roll': 'gỏi cuốn', 'fresh roll': 'gỏi cuốn',
    'fried roll': 'chả giò', 'egg roll': 'chả giò',
    'steamed roll': 'bánh cuốn', 'dumpling': 'há cảo', 'dimsum': 'dimsum',
    'snack': 'ăn vặt', 'street food': 'vỉa hè',
    
    # -- Lẩu & Nướng --
    'hotpot': 'lẩu', 'thai hotpot': 'lẩu thái',
    'bbq': 'nướng', 'grilled': 'nướng', 'steak': 'bít tết', 'beefsteak': 'bít tết',
    
    # -- Nguyên liệu --
    'seafood': 'hải sản', 'fish': 'cá', 'crab': 'cua', 'shrimp': 'tôm', 
    'snail': 'ốc', 'clam': 'nghêu', 'oyster': 'hàu',
    'beef': 'bò', 'chicken': 'gà', 'pork': 'heo', 'duck': 'vịt', 'goat': 'dê',
    'vegetarian': 'chay', 'vegan': 'chay', 'tofu': 'đậu hũ',
    
    # -- Đồ uống & Tráng miệng --
    'coffee': 'cà phê', 'milk coffee': 'cà phê sữa', 'egg coffee': 'cà phê trứng',
    'tea': 'trà', 'milk tea': 'trà sữa', 'bubble tea': 'trà sữa',
    'juice': 'nước ép', 'smoothie': 'sinh tố', 'beer': 'bia',
    'dessert': 'tráng miệng', 'sweet soup': 'chè', 'ice cream': 'kem', 'cake': 'bánh ngọt',
    
    # -- Tính chất --
    'delicious': 'ngon', 'yummy': 'ngon', 'tasty': 'ngon', 'good': 'ngon', 'best': 'ngon nhất',
    'cheap': 'rẻ', 'budget': 'rẻ', 'reasonable': 'rẻ', 'affordable': 'rẻ',
    'expensive': 'sang trọng', 'luxury': 'sang trọng', 'fine dining': 'sang trọng',
    'near': 'gần', 'nearby': 'gần', 'closest': 'gần',
    'spicy': 'cay', 'hot': 'cay',
    'nice view': 'đẹp', 'air conditioner': 'máy lạnh', 'ac': 'máy lạnh',
    
    # -- Thời gian & Địa điểm --
    'late night': 'ăn đêm', 'night': 'đêm', 'midnight': 'đêm',
    'morning': 'sáng', 'breakfast': 'sáng',
    'lunch': 'trưa', 'noon': 'trưa',
    'dinner': 'tối',
    'district': 'quận', 'city': 'thành phố', 'hcmc': 'tphcm', 'saigon': 'sài gòn'
}

LOCATION_NAMES = {
    # --- TP. HỒ CHÍ MINH ---
    'quận 1': 'Quận 1', 'q1': 'Quận 1', 
    'quận 2': 'Quận 2', 'q2': 'Quận 2',
    'quận 3': 'Quận 3', 'q3': 'Quận 3', 
    'quận 4': 'Quận 4', 'q4': 'Quận 4',
    'quận 5': 'Quận 5', 'q5': 'Quận 5', 
    'quận 6': 'Quận 6', 'q6': 'Quận 6',
    'quận 7': 'Quận 7', 'q7': 'Quận 7', 
    'quận 8': 'Quận 8', 'q8': 'Quận 8',
    'quận 9': 'Quận 9', 'q9': 'Quận 9', 
    'quận 10': 'Quận 10', 'q10': 'Quận 10',
    'quận 11': 'Quận 11', 'q11': 'Quận 11', 
    'quận 12': 'Quận 12', 'q12': 'Quận 12',
    'bình thạnh': 'Bình Thạnh', 'bt': 'Bình Thạnh',
    'phú nhuận': 'Phú Nhuận', 'pn': 'Phú Nhuận',
    'gò vấp': 'Gò Vấp', 'gv': 'Gò Vấp',
    'tân bình': 'Tân Bình', 'tb': 'Tân Bình',
    'tân phú': 'Tân Phú', 'tp': 'Tân Phú',
    'bình tân': 'Bình Tân', 
    'thủ đức': 'Thủ Đức', 'tđ': 'Thủ Đức',
    'sài gòn': 'TPHCM', 'hcm': 'TPHCM', 'tphcm': 'TPHCM',
    'bình chánh': 'Bình Chánh', 'hóc môn': 'Hóc Môn', 
    'củ chi': 'Củ Chi', 'nhà bè': 'Nhà Bè', 'cần giờ': 'Cần Giờ',

    # --- HÀ NỘI ---
    'hà nội': 'Hà Nội', 'hn': 'Hà Nội', 'thủ đô': 'Hà Nội',
    'ba đình': 'Ba Đình', 'bđ': 'Ba Đình',
    'hoàn kiếm': 'Hoàn Kiếm', 'hk': 'Hoàn Kiếm',
    'tây hồ': 'Tây Hồ', 
    'long biên': 'Long Biên', 'lb': 'Long Biên',
    'cầu giấy': 'Cầu Giấy', 'cg': 'Cầu Giấy',
    'đống đa': 'Đống Đa', 'đđ': 'Đống Đa',
    'hai bà trưng': 'Hai Bà Trưng', 'hbt': 'Hai Bà Trưng',
    'hoàng mai': 'Hoàng Mai', 'hm': 'Hoàng Mai',
    'thanh xuân': 'Thanh Xuân', 'tx': 'Thanh Xuân',
    'sóc sơn': 'Sóc Sơn', 'đông anh': 'Đông Anh', 'gia lâm': 'Gia Lâm',
    'nam từ liêm': 'Nam Từ Liêm', 'ntl': 'Nam Từ Liêm',
    'bắc từ liêm': 'Bắc Từ Liêm', 'btl': 'Bắc Từ Liêm',
    'hà đông': 'Hà Đông', 'hđ': 'Hà Đông',
    'sơn tây': 'Sơn Tây', 'ba vì': 'Ba Vì', 'phúc thọ': 'Phúc Thọ',
    'đan phượng': 'Đan Phượng', 'hoài đức': 'Hoài Đức', 'quốc oai': 'Quốc Oai',
    'thạch thất': 'Thạch Thất', 'chương mỹ': 'Chương Mỹ', 'thanh oai': 'Thanh Oai',
    'thường tín': 'Thường Tín', 'phú xuyên': 'Phú Xuyên', 'ứng hòa': 'Ứng Hòa',
    'mỹ đức': 'Mỹ Đức', 'mê linh': 'Mê Linh', 'thanh trì': 'Thanh Trì',

    # --- ĐÀ NẴNG ---
    'đà nẵng': 'Đà Nẵng', 'đn': 'Đà Nẵng', 'dn': 'Đà Nẵng',
    'hải châu': 'Hải Châu', 'hc': 'Hải Châu',
    'thanh khê': 'Thanh Khê', 'tk': 'Thanh Khê',
    'sơn trà': 'Sơn Trà', 'st': 'Sơn Trà',
    'ngũ hành sơn': 'Ngũ Hành Sơn', 'nhs': 'Ngũ Hành Sơn',
    'liên chiểu': 'Liên Chiểu', 'lc': 'Liên Chiểu',
    'cẩm lệ': 'Cẩm Lệ', 'cl': 'Cẩm Lệ',
    'hòa vang': 'Hòa Vang', 'hoàng sa': 'Hoàng Sa',
}

# [DANH SÁCH THÀNH PHỐ LỚN] - Dùng để lọc rộng hơn
MAJOR_CITIES = ['Hà Nội', 'TPHCM', 'Đà Nẵng']

candidate_tags = [
    # -- Món Nước --
    'bún bò', 'bún bò huế', 'bún riêu', 'bún mắm', 'bún chả', 'bún thịt nướng', 'bún đậu', 
    'bún cá', 'bún mọc', 'bún thái', 'bún ốc', 'bún',
    'phở', 'phở bò', 'phở gà', 'phở cuốn',
    'hủ tiếu', 'hủ tiếu nam vang', 'hủ tiếu gõ', 'hủ tiếu mực',
    'bánh canh', 'bánh canh cua', 'bánh canh ghẹ', 'bánh canh cá lóc',
    'mì', 'mì quảng', 'mì vịt tiềm', 'mì ý', 'mì cay', 'mì xào', 'mì trộn', 'ramen', 'udon',
    'miến', 'miến gà', 'miến lươn', 'miến xào',
    'nui', 'nui xào', 'bò kho', 'lagu', 'cà ri', 'cháo', 'cháo lòng', 'cháo ếch', 'súp',

    # -- Cơm --
    'cơm tấm', 'cơm sườn', 'cơm gà', 'cơm gà xối mỡ', 'cơm niêu', 'cơm văn phòng', 
    'cơm chiên', 'cơm rang', 'cơm lam', 'cơm phần', 'cơm',
    'xôi', 'xôi gà', 'xôi mặn',
    
    # -- Món Mặn / Nhậu --
    'lẩu', 'lẩu thái', 'lẩu bò', 'lẩu gà', 'lẩu dê', 'lẩu hải sản', 'lẩu mắm', 'lẩu cá',
    'nướng', 'bbq', 'bò nướng', 'gà nướng', 'hải sản nướng', 'nem nướng',
    'bít tết', 'bò né', 'bò bít tết', 'steak',
    'hải sản', 'ốc', 'tôm', 'cua', 'ghẹ', 'hàu', 'mực', 'bạch tuộc',
    'gà rán', 'gà luộc', 'gà ủ muối', 'vịt quay', 'heo quay', 'phá lấu',
    'dê', 'cừu', 'ếch', 'lươn',
    
    # -- Bánh & Ăn vặt --
    'bánh mì', 'bánh mì chảo', 'bánh mì xíu mại',
    'bánh xèo', 'bánh khọt', 'bánh cuốn', 'bánh ướt', 'bánh bèo', 'bánh bột lọc', 'bánh nậm',
    'gỏi cuốn', 'bì cuốn', 'chả giò', 'nem rán',
    'pizza', 'hamburger', 'sushi', 'sashimi', 'dimsum', 'há cảo', 'xíu mại',
    'ăn vặt', 'bánh tráng trộn', 'cá viên chiên', 'xiên que', 'bắp xào', 'hột vịt lộn',
    
    # -- Đồ uống & Tráng miệng --
    'cà phê', 'cà phê sữa', 'cà phê trứng', 'cà phê vợt',
    'trà sữa', 'trà đào', 'trà chanh', 'trà',
    'sinh tố', 'nước ép', 'chè', 'kem', 'bingsu', 'tàu hũ', 'sữa chua', 'bánh ngọt',
    'bia', 'bia thủ công', 'rượu', 'pub', 'bar',
    
    # -- Phong cách / Quốc gia --
    'chay', 'thuần chay', 'healthy', 'eat clean',
    'hàn quốc', 'nhật bản', 'trung hoa', 'thái lan', 'âu', 'mỹ', 'ý',
    'vỉa hè', 'sang trọng', 'bình dân', 'gia đình', 'hẹn hò', 'nhậu', 'view đẹp',
    'máy lạnh', 'sân vườn', 'yên tĩnh', 'nhanh', 'mang đi', 'buffet',
    
    # -- Thời gian --
    'sáng', 'trưa', 'chiều', 'tối', 'đêm', 'ăn đêm', '24h'
]

PRIORITY_TAGS = ['đêm', 'sáng', 'trưa', 'chiều', 'tối', 'ăn đêm']
ADJECTIVE_TAGS = ['rẻ', 'gần', 'ngon', 'tốt', 'nhanh', 'đẹp', 'vỉa hè', 'sang trọng', 'yên tĩnh', 'nổi tiếng', 'nhất']

# --- HELPER FUNCTIONS ---
def clean_coordinate(val):
    try:
        if isinstance(val, (int, float)): return float(val)
        if isinstance(val, str): return float(val.replace(',', '.'))
        return 0.0
    except: return 0.0

def clean_price(val):
    try:
        s = str(val)
        clean_s = ''.join(filter(str.isdigit, s))
        if clean_s:
            return float(clean_s)
        return 0.0
    except:
        return 0.0

# [FIX] Sửa lại hàm trích xuất địa chỉ: Bỏ điều kiện 'quận' và ưu tiên chuỗi dài nhất
def extract_district_from_address(address):
    if not isinstance(address, str): return ''
    addr_lower = address.lower()
    
    # Sắp xếp keys theo độ dài giảm dần để ưu tiên từ khóa dài
    # VD: Nếu địa chỉ là "Quận 12", nó sẽ khớp "Quận 12" trước thay vì "Quận 1"
    sorted_locations = sorted(LOCATION_NAMES.keys(), key=len, reverse=True)
    
    for k in sorted_locations:
        # Chỉ cần tên địa danh nằm trong địa chỉ là lấy (Bỏ điều kiện 'quận' in k)
        if k in addr_lower:
            return LOCATION_NAMES[k]
            
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
    global df, tfidf_vectorizer, sentiment_pipeline, yolo_model
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
            
            if 'price' not in temp_df.columns:
                temp_df['price'] = 0.0
            else:
                temp_df['price'] = temp_df['price'].apply(clean_price)
                
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

    try:
        print("-> Đang tải và khởi tạo YOLOv11m (Medium)...")
        yolo_model = YOLO("yolo11m.pt")
        print("-> Load YOLO model thành công!")
    except Exception as e:
        print(f"!!! Lỗi load YOLO: {e}")
        yolo_model = None

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
    
    if user_gps and len(user_gps) == 2:
        current_df['distance_km'] = current_df.apply(lambda row: calculate_distance(user_gps[0], user_gps[1], row['lat'], row['lon']), axis=1)
    else:
        current_df['distance_km'] = 999.0
        
    temp_query = " " + processed_query.lower() + " "
    
    for k, v in TAG_KNOWLEDGE_BASE.items():
        if f" {k} " in temp_query: temp_query = temp_query.replace(f" {k} ", f" {v} ")

    locations_to_filter = []
    for k, v in LOCATION_NAMES.items():
        if f" {k} " in temp_query:
            locations_to_filter.append(v)
            temp_query = temp_query.replace(f" {k} ", " ")
    
    extracted_tags = []
    sorted_candidates = sorted(candidate_tags, key=len, reverse=True)
    for tag in sorted_candidates:
        if f" {tag} " in temp_query:
            extracted_tags.append(tag)
            temp_query = temp_query.replace(f" {tag} ", " ")
            
    mandatory_tags = [t for t in extracted_tags if t in PRIORITY_TAGS]
    
    # --- BƯỚC 1: LỌC CƠ BẢN (Location, GPS) ---
    filtered_df = current_df
    
    if locations_to_filter:
        # [FIX] Logic thông minh hơn cho việc lọc địa điểm
        # Tách ra: Đâu là Tên thành phố, đâu là tên Quận huyện
        city_filters = [loc for loc in locations_to_filter if loc in MAJOR_CITIES]
        district_filters = [loc for loc in locations_to_filter if loc not in MAJOR_CITIES]

        if city_filters:
            # Nếu user tìm "Hà Nội" -> Tìm trong toàn bộ địa chỉ (vì quận Ba Đình cũng thuộc HN)
            pattern = '|'.join(city_filters)
            filtered_df = filtered_df[filtered_df['address'].str.contains(pattern, case=False, na=False)]
        
        if district_filters:
             # Nếu user tìm "Ba Đình" -> Tìm chính xác trong cột district đã extract
             filtered_df = filtered_df[filtered_df['district'].isin(district_filters)]

    elif request_data.city_filter:
        filtered_df = filtered_df[filtered_df['district'].str.contains(request_data.city_filter, case=False, na=False)]
    else:
        if user_gps and len(user_gps) == 2:
            filtered_df = filtered_df[filtered_df['distance_km'] <= 20.0]
            
    # --- BƯỚC 2: LỌC THỜI GIAN ---
    if mandatory_tags:
        for tag in mandatory_tags:
            filtered_df = filtered_df[filtered_df['tags'].str.contains(tag, case=False, na=False)]
            
    if filtered_df.empty:
        return {"sort_by": "relevance", "scores": []}
        
    # --- BƯỚC 3: LỌC MÓN ĂN ---
    target_dish_tags = [t for t in extracted_tags if t not in ADJECTIVE_TAGS and t not in PRIORITY_TAGS]
    
    if target_dish_tags:
        matches = []
        for index, row in filtered_df.iterrows():
            row_tags = str(row['tags']).lower()
            if any(dish in row_tags for dish in target_dish_tags):
                matches.append(index)
        
        if len(matches) > 0:
            filtered_df = filtered_df.loc[matches]

    # --- BƯỚC 4: CHẤM ĐIỂM ---
    final_query_text = " ".join(extracted_tags) if extracted_tags else processed_query
    
    try:
        qv = tfidf_vectorizer.transform([final_query_text])
        tm = tfidf_vectorizer.transform(filtered_df['tags'])
        base_scores = cosine_similarity(qv, tm).flatten()
    except:
        base_scores = [0.0] * len(filtered_df)

    # --- LOGIC BOOSTING ---
    boosted_scores = []
    name_query = processed_query.lower().strip() 

    for idx, row in enumerate(filtered_df.itertuples()):
        score = base_scores[idx]
        row_name = str(row.name).lower()
        
        if target_dish_tags:
            for dish in target_dish_tags:
                if dish in row_name:
                    score += 0.3 

        if len(name_query) > 2:
            if name_query == row_name:
                score += 10.0
            elif name_query in row_name:
                score += 5.0
        
        boosted_scores.append(score)

    filtered_df = filtered_df.copy()
    filtered_df['S_taste'] = boosted_scores

    final_candidates = filtered_df

    # --- BƯỚC 5: SẮP XẾP ---
    sort_by = "relevance"
    query_lower = processed_query.lower()
    
    if "gần" in query_lower:
        sort_by = "distance"
        final_candidates = final_candidates.sort_values('distance_km', ascending=True)
    elif "rẻ" in query_lower:
        sort_by = "price"
        final_candidates = final_candidates.sort_values('price', ascending=True)
    elif any(x in query_lower for x in ['ngon', 'tốt nhất', 'nổi tiếng', 'rating', 'đánh giá']):
        sort_by = "rating"
        final_candidates = final_candidates.sort_values('rating', ascending=False)
    else:
        final_candidates = final_candidates.sort_values('S_taste', ascending=False)
        
    scores_list = []
    for _, row in final_candidates.head(64).iterrows():
        scores_list.append({
            "id": str(row['id']),
            "name": str(row['name']),
            "tags": str(row['tags']),
            "S_taste": float(row.get('S_taste', 0.0)),
            "distance_km": float(row.get('distance_km', 999.0)),
            "price": int(row['price'])
        })
        
    return {"sort_by": sort_by, "scores": scores_list}

@app.post("/sentiment", response_model=SentimentResponse)
async def handle_sentiment(request_data: SentimentRequest):
    if not sentiment_pipeline: 
        return {"label": "neutral", "score": 0.5} 
    res = sentiment_pipeline(request_data.review)[0]
    return {"label": res['label'], "score": res['score']}

@app.post("/predict-food")
async def predict_food(file: UploadFile = File(...)):
    if not yolo_model:
        raise HTTPException(status_code=503, detail="Model AI chưa sẵn sàng")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        results = yolo_model(image)
        
        detected_name = ""
        max_conf = 0.0
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes:
                top_box = sorted(result.boxes, key=lambda x: x.conf, reverse=True)[0]
                max_conf = float(top_box.conf)
                cls_id = int(top_box.cls)
                detected_name = result.names[cls_id] 

        if not detected_name:
            return {"food_name": None, "message": "Không nhận diện được món ăn"}

        name_lower = detected_name.lower().replace("_", " ") 
        translated_name = EN_VI_MAPPING.get(name_lower, name_lower)

        if translated_name == name_lower: 
             for k, v in EN_VI_MAPPING.items():
                 if name_lower in k:
                     translated_name = v
                     break

        print(f"AI Detected: {detected_name} -> {translated_name}")

        return {
            "food_name": translated_name, 
            "original_name": detected_name,
            "confidence": max_conf
        }

    except Exception as e:
        print(f"Lỗi dự đoán ảnh: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=5000, reload=True)