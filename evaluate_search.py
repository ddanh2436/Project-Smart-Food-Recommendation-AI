import os

import requests
import json
import time
import statistics

# --- CẤU HÌNH ---
# Override to compare a local build against the deployed one:
#   AI_EVAL_URL=https://<space>.hf.space python evaluate_search.py
API_URL = os.getenv("AI_EVAL_URL", "http://127.0.0.1:5000").rstrip("/") + "/recommend"

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

TEST_CASES = {
    # 1. Test Món Ăn Đặc Thù (Có trong DB)
    "bún đậu mắm tôm": ["bún đậu", "mắm tôm"],
    "lẩu gà lá é phú yên": ["lẩu gà", "lá é", "phú yên"], # Test quán Con Gà Trống, Quán A7
    "cơm niêu singapore": ["cơm niêu", "thiên lý", "phương nam"],
    "bún quậy kiến xây": ["bún quậy", "saigon"], # Test Bún Quậy Saigon
    "nem nướng nha trang": ["nem nướng", "nha trang"], # Test Chị Liên, ABMA
    "bánh tằm cay cà mau": ["bánh tằm", "cà mau"], # Test quán 69
    "gà nướng cơm lam tây nguyên": ["gà nướng", "cơm lam", "tây nguyên", "bản đôn"],

    # 2. Test Vùng Miền & Phong Cách
    "món ngon hà nội": ["hà nội", "món bắc", "bún chả", "phở"],
    "đặc sản huế": ["huế","cố đô"],
    "món ăn miền tây dân dã": ["miền tây", "bún mắm", "bánh xèo", "cà mau"],
    "ẩm thực tây bắc": ["tây bắc", "mẹt", "heo quay"], # Test Mr Nhoi, Men Quán

    # 3. Test Theo Địa Điểm (Rất quan trọng với dữ liệu của bạn)
    "quán ăn ngon quận 4": ["quận 4", "bún mắm", "ốc", "phở"], # DB bạn có rất nhiều quán Q4
    "nhà hàng thảo điền quận 2": ["thảo điền", "quận 2", "sang trọng"],
    "ăn uống khu phan xích long": ["hoa phượng", "hoa lan", "phú nhuận"], # Test khu vực Phú Nhuận

    # 4. Test Theo Nhu Cầu/Không Gian (Tags)
    "quán nhậu có máy lạnh": ["nhậu", "máy lạnh", "lẩu", "nướng"],
    "ăn sáng món nước": ["phở", "hủ tiếu", "bún", "ăn sáng"],
    "ăn đêm vỉa hè": ["ăn đêm", "vỉa hè", "cháo", "ốc"],
    "nhà hàng sang trọng tiếp khách": ["sang trọng", "tiếp khách", "doanh nhân"], # Test Hương Lúa 8, Chạn Bếp

    # 5. Test Món "Hiếm" hoặc Từ Khóa Ngách (Kiểm tra độ nhạy)
    "bún bò sụn": ["bún bò", "sụn"],
    "heo quay tây bắc": ["heo quay", "tây bắc"],
    "cháo lòng": ["cháo lòng", "dồi"],
    "nui xào bò": ["nui xào", "cô hai"],
}

def check_relevance(result_item, expected_keywords):
    """Kiểm tra xem kết quả có chứa từ khóa mong đợi không"""
    # Gộp tên quán và tags lại để tìm kiếm
    text_to_search = (result_item.get('name', '') + " " + result_item.get('tags', '')).lower()
    
    for keyword in expected_keywords:
        if keyword.lower() in text_to_search:
            return True, keyword # Trả về True và từ khóa tìm thấy
    return False, None

def run_evaluation(top_k=5):
    print(f"{Colors.HEADER}--- BẮT ĐẦU KIỂM THỬ AI SEARCH (TOP {top_k}) ---{Colors.ENDC}\n")
    
    total_cases = len(TEST_CASES)
    passed_cases = 0
    latencies = []
    
    for query, expected_tags in TEST_CASES.items():
        print(f"Testing: {Colors.OKBLUE}'{query}'{Colors.ENDC}...", end=" ")
        
        start_time = time.time()
        try:
            # Gửi request giả lập
            response = requests.post(API_URL, json={"query": query})
            latency = (time.time() - start_time) * 1000 # đổi ra ms
            latencies.append(latency)
            
            if response.status_code != 200:
                print(f"{Colors.FAIL}[ERROR API]{Colors.ENDC}")
                continue
                
            data = response.json()
            results = data.get("scores", [])[:top_k] # Lấy Top K kết quả đầu tiên
            
            if not results:
                print(f"{Colors.WARNING}[NO RESULT]{Colors.ENDC} - AI không tìm thấy gì")
                continue

            # Đánh giá: Trong Top K quán, có quán nào phù hợp không?
            is_relevant = False
            matched_keyword = ""
            best_match_name = ""
            
            for item in results:
                found, keyword = check_relevance(item, expected_tags)
                if found:
                    is_relevant = True
                    matched_keyword = keyword
                    best_match_name = item['name']
                    break # Chỉ cần tìm thấy 1 quán đúng trong Top K là coi như Pass
            
            if is_relevant:
                passed_cases += 1
                print(f"{Colors.OKGREEN}[PASS]{Colors.ENDC} ({latency:.0f}ms) -> Khớp: '{matched_keyword}' tại quán '{best_match_name}'")
            else:
                print(f"{Colors.FAIL}[FAIL]{Colors.ENDC} ({latency:.0f}ms)")
                print(f"   -> Mong đợi: {expected_tags}")
                print(f"   -> Thực tế Top 1: {results[0]['name']} | Tags: {results[0]['tags']}")

        except Exception as e:
            print(f"{Colors.FAIL}[EXCEPTION] {e}{Colors.ENDC}")

    # --- TỔNG KẾT ---
    accuracy = (passed_cases / total_cases) * 100
    avg_latency = statistics.mean(latencies) if latencies else 0
    
    print("\n" + "="*40)
    print(f"{Colors.HEADER}KẾT QUẢ ĐÁNH GIÁ{Colors.ENDC}")
    print("="*40)
    print(f"Tổng số test case: {total_cases}")
    print(f"Số case ĐẠT:       {Colors.OKGREEN}{passed_cases}{Colors.ENDC}")
    print(f"Số case HỎNG:      {Colors.FAIL}{total_cases - passed_cases}{Colors.ENDC}")
    print(f"Độ chính xác:      {Colors.OKBLUE}{accuracy:.2f}%{Colors.ENDC}")
    print(f"Độ trễ trung bình: {avg_latency:.1f} ms")
    print("="*40)

    if accuracy >= 80:
        print(f"{Colors.OKGREEN}ĐÁNH GIÁ: Model hoạt động TỐT! 🚀{Colors.ENDC}")
    elif accuracy >= 50:
        print(f"{Colors.WARNING}ĐÁNH GIÁ: Model KHÁ, cần cải thiện data synonyms.{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}ĐÁNH GIÁ: Model YẾU, cần kiểm tra lại code hoặc DB.{Colors.ENDC}")

if __name__ == "__main__":
    # Cần cài thư viện requests: pip install requests
    run_evaluation(top_k=5)