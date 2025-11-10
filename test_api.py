import requests
import json
import time

# -----------------------------------------------------------------
# CÀI ĐẶT (CONFIG)
# -----------------------------------------------------------------
URL_RECOMMEND = "http://127.0.0.1:5000/recommend"
URL_SENTIMENT = "http://127.0.0.1:5000/sentiment"

# Giả lập NestJS đã lọc GPS (20km) và tìm được 20 quán đầu tiên
# (Dùng list(range(1, 21)) để lấy 20 ID đầu tiên từ file .csv)
GIA_LAP_20_QUAN_GAN = list(range(1, 21)) 

# -----------------------------------------------------------------
# HÀM TEST (ĐỂ GỌI API)
# -----------------------------------------------------------------

# Hàm test /recommend
def test_recommend(test_name, query, candidate_ids):
    print(f"\n--- {test_name} ---")
    print(f"Đang gửi Query: '{query}'")
    
    payload_rec = {
        "query": query, 
        "candidate_ids": candidate_ids
    }
    
    try:
        response_rec = requests.post(URL_RECOMMEND, json=payload_rec)
        print(f"Server trả về (Status Code): {response_rec.status_code}")
        
        if response_rec.status_code == 200:
            print("Dữ liệu (S_taste & Sort_by) server trả về:")
            print(json.dumps(response_rec.json(), indent=2, ensure_ascii=False))
        else:
            print(f"LỖI TỪ SERVER: {response_rec.text}")
            
    except Exception as e:
        print(f"LỖI KẾT NỐI: Không kết nối được. Server 'api_fast.py' đã chạy chưa? Lỗi: {e}")
    
    print("-" * (len(test_name) + 6))
    time.sleep(0.5) # Chờ 0.5s để xem log cho dễ

# Hàm test /sentiment
def test_sentiment(test_name, review):
    print(f"\n--- {test_name} ---")
    print(f"Đang gửi Review: '{review}'")
    
    payload_sen = {"review": review}
    
    try:
        response_sen = requests.post(URL_SENTIMENT, json=payload_sen)
        print(f"Server trả về (Status Code): {response_sen.status_code}")
        
        if response_sen.status_code == 200:
            print("Dữ liệu (Sentiment) server trả về:")
            print(json.dumps(response_sen.json(), indent=2, ensure_ascii=False))
        else:
            print(f"LỖI TỪ SERVER: {response_sen.text}")
            
    except Exception as e:
        print(f"LỖI KẾT NỐI: {e}")
    
    print("-" * (len(test_name) + 6))
    time.sleep(0.5)

# -----------------------------------------------------------------
# CHẠY TEST
# -----------------------------------------------------------------
if __name__ == "__main__":
    
    print("==============================================")
    print("BẮT ĐẦU TEST SERVER AI (100% TỰ HOST)...")
    print("==============================================")

    # --- KỊCH BẢN 1: TEST LỚP 1+2 ("Bún bò rẻ") ---
    # Mong đợi: sort_by: 'price', và 'scores' chứa ID 1 (Bún Bò O Xinh)
    test_recommend(
        "TEST 1: Lớp 1 (Sort_by) + Lớp 2 (Tag)",
        "tôi muốn ăn bún bò rẻ",
        GIA_LAP_20_QUAN_GAN
    )

    # --- KỊCH BẢN 2: TEST LỚP 3 ("Đồng quê") ---
    # Mong đợi: sort_by: 'taste', và 'scores' chứa ID 20 (Cục Gạch Quán)
    test_recommend(
        "TEST 2: Lớp 3 (PhoBERT Fallback)",
        "món ăn đồng quê Việt Nam",
        GIA_LAP_20_QUAN_GAN
    )

    # --- KỊCH BẢN 3: TEST LỚP 1 ("Huế") ---
    # Mong đợi: sort_by: 'taste', và 'scores' chứa ID 1, 8, 5 (các quán miền Trung)
    test_recommend(
        "TEST 3: Lớp 1 (Knowledge Base - 'Huế')",
        "thèm đặc sản Huế",
        GIA_LAP_20_QUAN_GAN
    )

    # --- KỊCH BẢN 4: TEST LỚP 1 ("Lãng mạn") ---
    # Mong đợi: sort_by: 'taste', và 'scores' chứa ID 4, 13 (Pizza 4P, Sushi)
    test_recommend(
        "TEST 4: Lớp 1 (Knowledge Base - 'Lãng mạn')",
        "quán nào lãng mạn cho 2 người",
        GIA_LAP_20_QUAN_GAN
    )

    # --- KỊCH BẢN 5: TEST /sentiment (5CD-AI) ---
    # Mong đợi: {'label': 'POS', ...}
    test_sentiment(
        "TEST 5: /sentiment (Câu phủ định)",
        "Quán này không dở đâu."
    )
    
    print("\n==============================================")
    print("TEST HOÀN TẤT.")
    print("==============================================")