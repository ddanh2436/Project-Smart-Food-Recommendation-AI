import requests
import json
import time

# --- CẤU HÌNH ---
URL_RECOMMEND = "http://127.0.0.1:5000/recommend"

# GIẢ LẬP: GPS Của người dùng (Ví dụ: Đang đứng ở Chợ Bến Thành, Q1, TPHCM)
CURRENT_USER_GPS = [10.7721, 106.6983] 

def call_ai_server(user_query):
    print(f"\n{'='*60}")
    print(f"🔍 Đang tìm: '{user_query}'")
    print(f"📍 Vị trí hiện tại: TP. Hồ Chí Minh (GPS: {CURRENT_USER_GPS})")
    print(f"{'='*60}")

    # Gửi request đơn giản nhất: Chỉ Query + GPS
    payload = {
        "query": user_query,
        "candidate_ids": [], 
        "user_gps": CURRENT_USER_GPS
    }

    try:
        start_time = time.time()
        response = requests.post(URL_RECOMMEND, json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            sort_by = data['sort_by']
            scores = data['scores']
            
            print(f"✅ Tìm thấy {len(scores)} kết quả (trong {end_time - start_time:.3f}s)")
            print(f"🎯 Tiêu chí xếp hạng AI chọn: '{sort_by.upper()}'")
            
            if not scores:
                print("\n   (Không tìm thấy quán nào phù hợp)")
                return

            # Lấy Top 5
            top_n = scores[:5]
            
            print(f"\n--- DANH SÁCH ĐỀ XUẤT ---")
            for i, item in enumerate(top_n):
                if item['distance_km'] > 1000:
                    dist_str = "Cách rất xa (>1000km)" 
                else:
                    dist_str = f"Cách {item['distance_km']:.2f} km"
                
                price_str = f"{item.get('price', 0):,}đ"
                
                print(f"#{i+1} [ID:{item['id']}] {item['name'].upper()}")
                print(f"   📍 {dist_str} | 💵 {price_str} | ⭐ Taste: {item['S_taste']:.2f}")
                print(f"   🏷️  Tags: {item['tags']}")
                print("-" * 40)
                
        else:
            print(f"❌ LỖI SERVER: {response.text}")

    except Exception as e:
        print(f"❌ LỖI KẾT NỐI: {e}")
        print("   (Hãy chắc chắn bạn đã chạy 'py -m uvicorn api:app --reload' ở cửa sổ kia)")

# --- VÒNG LẶP CHÍNH ---
def main():
    print("\n*************************************************")
    print("   APP DU LỊCH THÔNG MINH (SEARCH BAR DEMO)")
    print("   (Gõ 'exit' để thoát)")
    print("*************************************************")

    while True:
        # Thanh tìm kiếm duy nhất
        query = input("\n🔎 Bạn muốn ăn gì hôm nay? > ").strip()
        
        if query.lower() in ['exit', 'quit', 'thoat']:
            print("Tạm biệt! Hẹn gặp lại.")
            break
        
        if not query:
            continue

        call_ai_server(query)

if __name__ == "__main__":
    main()