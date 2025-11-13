import requests
import json
import time

# -----------------------------------------------------------------
# CÀI ĐẶT (CONFIG)
# -----------------------------------------------------------------
URL_RECOMMEND = "http://127.0.0.1:5000/recommend"

# -----------------------------------------------------------------
# HÀM TEST (ĐỂ GỌI API)
# -----------------------------------------------------------------

# (Hàm này giả lập NestJS gọi api_fast.py, sau đó tự Lọc và Xếp hạng)
def run_full_flow(test_name, user_query, user_gps, candidate_ids_from_nest):
    print(f"\n==============================================")
    print(f"BẮT ĐẦU: {test_name}")
    print(f"==============================================")
    print(f"1. User (Giả lập): Gõ '{user_query}'")
    print(f"   (Đang ở {user_gps['city']})")
    print(f"2. NestJS (Giả lập): Lọc GPS {user_gps['city']} -> Tìm thấy {len(candidate_ids_from_nest)} quán gần.")
    
    # 3. GỌI API (api_fast.py)
    payload_rec = {
        "query": user_query, 
        "candidate_ids": candidate_ids_from_nest
    }
    
    try:
        response = requests.post(URL_RECOMMEND, json=payload_rec)
        if response.status_code != 200:
            print(f"LỖI TỪ SERVER: {response.text}")
            return
            
        data = response.json()
        
        # 4. NHẬN KẾT QUẢ TỪ SERVER AI (api_fast.py)
        sort_by = data['sort_by']
        scores_list = data['scores']
        
        print(f"3. Server AI (api_fast.py) đã chạy Pipeline 3 Lớp:")
        print(f"   -> Đã suy luận 'sort_by': '{sort_by}'")
        print(f"   -> Đã tính S_taste cho {len(scores_list)} quán.")

        # 5. NestJS (Giả lập) - LỌC CHUYÊN SÂU (Filter 2)
        # (Lọc các quán có S_taste > 0.5)
        final_candidates = [q for q in scores_list if q['S_taste'] > 0.4]
        
        if not final_candidates:
            print("\n--- KẾT QUẢ CUỐI CÙNG ---")
            print("Không tìm thấy nhà hàng nào phù hợp.")
            return

        print(f"4. NestJS (Giả lập): Lọc chuyên sâu S_taste > 0.4 -> Còn {len(final_candidates)} quán.")

        # 6. NestJS (Giả lập) - XẾP HẠNG (Rank)
        # (Sắp xếp theo S_taste để demo)
        final_ranking = sorted(final_candidates, key=lambda x: x['S_taste'], reverse=True)
        top_3 = final_ranking[:3] # Chỉ lấy Top 3

        print(f"5. NestJS (Giả lập): Xếp hạng theo '{sort_by}' (Demo: S_taste) -> Top 3:")
        
        print("\n--- KẾT QUẢ CUỐI CÙNG (TOP 3) ---")
        for i, restaurant in enumerate(top_3):
            print(f"#{i+1}: {restaurant['name']}")
            print(f"    (ID: {restaurant['id']}, S_taste: {restaurant['S_taste']:.4f}, Tags: {restaurant['tags'][:50]}...)")

    except Exception as e:
        print(f"LỖI KẾT NỐI: Không kết nối được. Server 'api_fast.py' đã chạy chưa? Lỗi: {e}")

# -----------------------------------------------------------------
# CHẠY 3 KỊCH BẢN TEST CỦA BẠN
# -----------------------------------------------------------------
if __name__ == "__main__":
    
    # --- Định nghĩa 2 "User" ---
    
    # User 1: Ở TP.HCM
    USER_HCM_GPS = {"city": "TPHCM", "coords": [10.77, 106.69]}
    # Giả lập NestJS lọc 20km ở TPHCM -> tìm được các quán (ID 1 -> 80)
    CANDIDATES_HCM = list(range(1, 81)) 
    
    # User 2: Ở Hà Nội
    USER_HANOI_GPS = {"city": "Hà Nội", "coords": [21.02, 105.83]}
    # Giả lập NestJS lọc 20km ở Hà Nội -> tìm được các quán (ID 81 -> 155)
    CANDIDATES_HANOI = list(range(81, 156)) 

    # --- Chạy 3 Kịch bản ---

    # KỊCH BẢN 1: KHÔNG NHẬP GÌ (Dùng GPS TP.HCM)
    # Mong đợi: api_fast.py chạy, trả về S_taste=0 cho query rỗng, kết quả 0 quán.
    run_full_flow(
        "Kịch bản 1: Query Rỗng (Mặc định TPHCM)",
        user_query="",
        user_gps=USER_HCM_GPS,
        candidate_ids_from_nest=CANDIDATES_HCM
    )

    # KỊCH BẢN 2: LỆNH VỊ TRÍ ("ở hà nội")
    # Mong đợi: api_fast.py (Lớp 1C) phát hiện "ở hà nội", BỎ QUA 75 quán TPHCM,
    #          tự lọc ra các quán Hà Nội (ID 76-150) và trả về.
    run_full_flow(
        "Kịch bản 2: Lệnh Vị trí (User ở TPHCM, tìm 'ở Hà Nội')",
        user_query="tôi muốn ăn phở ở hà nội", # (Lớp 1C sẽ bắt "ở hà nội")
        user_gps=USER_HCM_GPS,
        candidate_ids_from_nest=CANDIDATES_HCM # (Gửi 75 quán TPHCM)
    )

    # KỊCH BẢN 3: LỆNH SỞ THÍCH ("đặc sản hà nội" ở TPHCM)
    # Mong đợi: api_fast.py (Lớp 1A) dịch "đặc sản hà nội" -> "phở bún chả bắc",
    #          KHÔNG phát hiện Lệnh Vị trí,
    #          DÙNG 75 quán TPHCM (NestJS gửi), và tìm các quán (ID 2, 11) khớp "bắc".
    run_full_flow(
        "Kịch bản 3: Lệnh Sở thích (Tìm 'Vị' Hà Nội ở TPHCM)",
        user_query="tôi muốn ăn đặc sản hà nội", # (Lớp 1A sẽ bắt "đặc sản hà nội")
        user_gps=USER_HCM_GPS,
        candidate_ids_from_nest=CANDIDATES_HCM # (Gửi 75 quán TPHCM)
    )

    run_full_flow(
        "Kịch bản 4: Lệnh Sắp xếp (Tìm 'bún bò rẻ' ở TPHCM)",
        user_query="tôi muốn ăn bún bò rẻ", # (Lớp 1B bắt "rẻ", Lớp 2 bắt "bún bò")
        user_gps=USER_HCM_GPS,
        candidate_ids_from_nest=CANDIDATES_HCM # (Gửi 75 quán TPHCM)
    )