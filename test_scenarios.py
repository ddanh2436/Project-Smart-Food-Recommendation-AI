import requests
import json
import pandas as pd
import os

URL_RECOMMEND = "http://127.0.0.1:5000/recommend"

# --- HÀM GIẢ LẬP DATABASE CỦA NESTJS ---
# Đọc file CSV để tìm ID quán theo Thành phố hoặc Vùng miền
def get_candidate_ids_from_csv(filter_text):
    try:
        # Đọc file restaurants.csv (Giả lập Database)
        df = pd.read_csv('restaurants.csv')
        
        # Logic lọc: Kiểm tra xem Tags có chứa filter_text không
        # Tag đầu tiên là Thành phố (TPHCM, Hà Nội)
        # Tag thứ hai là Vùng miền (Miền Trung, Miền Bắc...)
        
        # Lọc các dòng mà cột 'tags' chứa filter_text
        filtered = df[df['tags'].str.contains(filter_text, case=False, na=False)]
        
        ids = filtered['id'].tolist()
        print(f"   [DB Simulator] Tìm thấy {len(ids)} quán khớp với '{filter_text}'")
        return ids
    except Exception as e:
        print(f"   [DB Simulator] Lỗi đọc CSV: {e}")
        return []

def run_test_scenario(test_name, user_query, user_gps, db_filter_criteria):
    print(f"\n{'='*50}")
    print(f"BẮT ĐẦU: {test_name}")
    print(f"{'='*50}")
    
    # 1. NestJS: Lọc Database sơ bộ
    print(f"1. NestJS: Lọc Database theo tiêu chí '{db_filter_criteria}'...")
    candidate_ids = get_candidate_ids_from_csv(db_filter_criteria)
    
    if not candidate_ids:
        print("   -> Không tìm thấy quán nào trong DB giả lập. Dừng test.")
        return

    # 2. NestJS: Gọi API AI
    print(f"2. NestJS: Gọi API AI với query '{user_query}'...")
    payload = {
        "query": user_query,
        "candidate_ids": candidate_ids,
        "user_gps": user_gps # Gửi thêm GPS để AI tính khoảng cách
    }
    
    try:
        response = requests.post(URL_RECOMMEND, json=payload)
        if response.status_code == 200:
            data = response.json()
            sort_by = data['sort_by']
            scores = data['scores']
            
            print(f"3. Server AI phản hồi: sort_by='{sort_by}', tìm thấy {len(scores)} quán phù hợp.")
            
            print("\n--- KẾT QUẢ HIỂN THỊ CHO USER (TOP 3) ---")
            for i, item in enumerate(scores[:3]):
                dist_info = f"{item['distance_km']:.2f} km" if item['distance_km'] < 900 else "N/A"
                print(f"#{i+1}: {item['name']} (ID: {item['id']})")
                print(f"    Taste: {item['S_taste']:.2f} | Dist: {dist_info} | Tags: {item['tags'][:60]}...")
        else:
            print(f"LỖI SERVER: {response.text}")
    except Exception as e:
        print(f"LỖI KẾT NỐI: {e}")

if __name__ == "__main__":
    # Tọa độ giả lập
    GPS_TPHCM = [10.77, 106.69] # Quận 1
    GPS_HANOI = [21.02, 105.83] # Hoàn Kiếm

    # --- KỊCH BẢN 1: Mặc định (Ở TPHCM, tìm bún bò) ---
    # Logic: Lọc DB lấy tất cả quán ở "TPHCM"
    run_test_scenario(
        "Kịch bản 1: User ở TPHCM, tìm 'bún bò' (Mặc định)",
        user_query="tôi muốn ăn bún bò",
        user_gps=GPS_TPHCM,
        db_filter_criteria="TPHCM" # Lọc Tag đầu tiên
    )

    # --- KỊCH BẢN 2: Đặc sản vùng miền (Ở TPHCM, tìm món Bắc) ---
    # Logic: Lọc DB lấy tất cả quán ở "TPHCM"
    # AI sẽ dịch "đặc sản hà nội" -> "bún chả, phở..." và tìm trong list TPHCM
    run_test_scenario(
        "Kịch bản 2: User ở TPHCM, tìm 'đặc sản hà nội'",
        user_query="tôi muốn ăn đặc sản hà nội",
        user_gps=GPS_TPHCM,
        db_filter_criteria="TPHCM" 
    )

    # --- KỊCH BẢN 3: Lệnh Vị trí (Tìm ở Hà Nội) ---
    # Logic: Lệnh "ở hà nội" -> NestJS (hoặc AI) chuyển hướng tìm quán có tag "Hà Nội"
    # Lưu ý: Trong thực tế NestJS sẽ xử lý việc này. Ở đây ta giả lập gửi candidate Hà Nội.
    run_test_scenario(
        "Kịch bản 3: User ở TPHCM, tìm 'phở ở hà nội'",
        user_query="tôi muốn ăn phở ở hà nội",
        user_gps=GPS_TPHCM, # GPS vẫn ở HCM
        db_filter_criteria="Hà Nội" # Giả lập NestJS đã hiểu và lọc quán Hà Nội
    )
    
    # --- KỊCH BẢN 4: Tìm theo khoảng cách (Gần đây nhất) ---
    run_test_scenario(
        "Kịch bản 4: User ở TPHCM, tìm 'quán chay gần đây nhất'",
        user_query="quán chay gần đây nhất",
        user_gps=GPS_TPHCM,
        db_filter_criteria="TPHCM"
    )