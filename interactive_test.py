import requests
import json
import time

URL_RECOMMEND = "http://127.0.0.1:5000/recommend"

GPS_HCM = [10.7762, 106.7009]
GPS_HANOI = [21.0285, 105.8542]

def call_ai_server(user_query, user_gps, city_context):
    print(f"\n[Client] Gửi query: '{user_query}' (Context: {city_context})")

    # CHỈ CẦN GỬI CITY FILTER
    payload = {
        "query": user_query,
        "city_filter": city_context, # Server sẽ tự lọc
        "candidate_ids": [], # Không cần ID nữa
        "user_gps": user_gps
    }

    try:
        start_time = time.time()
        response = requests.post(URL_RECOMMEND, json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            sort_by = data['sort_by']
            scores = data['scores']
            
            print(f"\n>>> KẾT QUẢ ({len(scores)} quán, Sort: {sort_by.upper()}, Time: {end_time - start_time:.2f}s):")
            
            top_3 = scores[:3]
            if not top_3:
                print("    Không tìm thấy quán nào.")
            else:
                for i, item in enumerate(top_3):
                    dist = f"{item['distance_km']:.2f} km" if item['distance_km'] < 900 else "N/A"
                    print(f"    #{i+1}: {item['name']} | Taste: {item['S_taste']:.2f} | Dist: {dist}")
                    print(f"        Tags: {item['tags'][:50]}...")
        else:
            print(f"LỖI SERVER: {response.text}")

    except Exception as e:
        print(f"LỖI KẾT NỐI: {e}")

def main():
    print("=== DEMO TƯƠNG TÁC (NO ID FILTER) ===")
    while True:
        print("\nChọn thành phố:")
        print("1. TP.HCM")
        print("2. Hà Nội")
        choice = input("Chọn (1/2): ").strip()
        
        if choice == '1':
            gps, city = GPS_HCM, "TPHCM"
        elif choice == '2':
            gps, city = GPS_HANOI, "Hà Nội"
        else: return

        while True:
            q = input(f"\n[{city}] Bạn muốn ăn gì? (Enter để đổi TP): ").strip()
            if not q: break
            call_ai_server(q, gps, city)

if __name__ == "__main__":
    main()