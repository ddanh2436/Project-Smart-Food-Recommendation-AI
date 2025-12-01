import requests
import json
import time

# Cấu hình
API_URL = "http://127.0.0.1:5000/sentiment"

# Danh sách các test case (Bao gồm nhiều trường hợp khó)
test_cases = [
    # --- Nhóm 1: Tích cực (Positive) ---
    "Món ăn ở đây ngon tuyệt vời, giá cả lại phải chăng.",
    "Nhân viên phục vụ rất chu đáo và nhiệt tình, sẽ quay lại.",
    "Không gian quán đẹp, thoáng mát, view xịn xò.",
    
    # --- Nhóm 2: Tiêu cực (Negative) ---
    "Đồ ăn dở tệ, thịt bị hôi, không bao giờ quay lại.",
    "Thái độ nhân viên lồi lõm, coi thường khách hàng.",
    "Chờ món quá lâu, hơn 30 phút mới có đồ ăn.",
    "Vệ sinh quá kém, thấy có con ruồi trong canh.",

    # --- Nhóm 3: Trung tính (Neutral) / Hỗn hợp (Mixed) ---
    "Món ăn tạm được nhưng giá hơi cao so với mặt bằng chung.",
    "Quán bình thường, không có gì đặc sắc lắm.",
    "Gửi xe hơi bất tiện nhưng đồ ăn thì ổn.",

    # --- Nhóm 4: Teencode / Tiếng lóng / Viết tắt (Thử thách AI) ---
    "quán này đồ ăn ngon vcl",  # Slang mạnh
    "chán ko tả nổi, phí tiền", # Viết tắt
    "đồ ăn cũng dc",            # Viết tắt
    "view dep nhung nuoc uong te", # Không dấu

    "Trước khi mình quyết định ra đây mua cũng đã tham khảo một số review trước trên foody chê quán bẩn, nhưng thật sự chưa bao giờ nghĩ đến cái mức kinh khủng như thế này. Khoan nói về sự bẩn, mình sẽ nói về khẩu vị cá nhân của mình, mình mua về 3 suất và 6 nem, xem ảnh thấy quán rán nem có vẻ khá chuẩn bắc nên qua mua thử về ăn xem như thế nào, thì xin thưa là bún chả HN, mình là ng HN và từ bé đến lớn mình chưa ăn cái bún chả và nem nào ở HN chán như ở cái chỗ này. bánh đa quá dày và ngấm dầu, dai, ăn cực ngấy, bún sợi to, chả thì như là rán lên chứ ko phải nướng. Và điều quan trọng nhất là ĐƯỢC ĂN BÚN CHẢ KÈM VỚI RUỒI (HOẶC BẤT KÌ MỘT CON KINH KHỦNG NÀO Ở ẢNH DƯỚI ?!?).Thật sự các bạn bán cái đống này với giá 198k kèm theo một con ruồi như thế này mà các bạn không thấy day dứt lương tâm à? ăn đc một miếng xong nhìn thấy con ruồi mà chỉ muốn nôn hết cái đống vừa nuốt trôi ra, quá là kinh hãi luôn và tạm biệt không hẹn ngày gặp lại nhé, làm ăn thế này thì nể các bạn quá rồi"
]

def run_test():
    print(f"--- BẮT ĐẦU TEST SENTIMENT ANALYSIS ({len(test_cases)} cases) ---\n")
    
    success_count = 0
    
    for i, review in enumerate(test_cases):
        payload = {"review": review}
        
        try:
            # Gửi request đến API
            start_time = time.time()
            response = requests.post(API_URL, json=payload)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000 # ms

            if response.status_code == 200:
                data = response.json()
                label = data.get('label')
                score = data.get('score', 0.0)
                
                # Mapping Label (Model này thường trả về LABEL_0, 1, 2)
                # LABEL_0: Tiêu cực (Negative)
                # LABEL_1: Trung tính (Neutral)
                # LABEL_2: Tích cực (Positive)
                human_label = label
                color = "\033[0m" # Reset
                
                if label == "LABEL_2" or label == "POS":
                    human_label = "TÍCH CỰC (Positive)"
                    color = "\033[92m" # Green
                elif label == "LABEL_0" or label == "NEG":
                    human_label = "TIÊU CỰC (Negative)"
                    color = "\033[91m" # Red
                elif label == "LABEL_1" or label == "NEU":
                    human_label = "TRUNG TÍNH (Neutral)"
                    color = "\033[93m" # Yellow

                print(f"Case {i+1}: {review}")
                print(f"   -> AI Đánh giá: {color}{human_label}\033[0m")
                print(f"   -> Độ tin cậy:  {score:.4f}")
                print(f"   -> Thời gian:   {latency:.2f}ms")
                print("-" * 50)
                success_count += 1
            else:
                print(f"Case {i+1}: LỖI {response.status_code} - {response.text}")

        except Exception as e:
            print(f"Case {i+1}: Lỗi kết nối - {e}")
            print("=> Bạn đã bật server 'python api.py' chưa?")
            break

    print(f"\n--- KẾT THÚC: Thành công {success_count}/{len(test_cases)} ---")

if __name__ == "__main__":
    run_test()