## 🗓️ NOTE AI - Cập nhật 13/11

Chào team,

Hôm nay tôi (phụ trách AI/Python) đã hoàn thành nâng cấp "bộ não" AI (`api_fast.py`) và file dữ liệu (`restaurants.csv`) để xử lý logic **Lọc theo Vị trí (Location Filtering)** một cách thông minh.

### 1. Cập nhật Dữ liệu (`restaurants.csv`)

* File `restaurants.csv` (tổng 155 nhà hàng) đã được "dọn dẹp" và chuẩn hóa ID.
* **TPHCM:** 80 quán, ID từ `1` đến `80`.
* **Hà Nội:** 75 quán, ID từ `81` đến `155`.
* **Quan trọng:** Cột `district` đã được **Trừu tượng hóa (Abstracted)**. Giờ đây nó chỉ chứa 2 giá trị duy nhất: `TPHCM` hoặc `Hà Nội` để phục vụ cho việc test logic.

### 2. Nâng cấp "Bộ não" AI (`api_fast.py`)

"Bộ não" (Pipeline 3 Lớp) giờ đã có thể phân biệt 2 kịch bản "Hà Nội" khác nhau:

* **Lệnh Lọc Vị trí:** User muốn tìm quán *tại* một nơi khác (ví dụ: `"ở hà nội"`).
* **Lệnh Lọc Sở thích:** User muốn tìm quán có *hương vị* của nơi khác, nhưng *ở gần* họ (ví dụ: `"đặc sản hà nội"` ở TPHCM).

---

### 3. Luồng hoạt động MỚI (Quan trọng cho NestJS)

Đây là 4 kịch bản chính mà `api_fast.py` (Server AI) giờ sẽ xử lý:

#### Kịch bản 1: Query Mặc định (Dùng GPS User)

* **User gõ:** `"tôi muốn ăn bún bò"`
* **NestJS (Backend):** Lọc 20km (dựa trên GPS User ở TPHCM) -> Tìm được 80 quán `candidate_ids` (ID 1-80).
* **`api_fast.py` (AI):**
    * Không phát hiện "Lệnh Vị trí" (vì không có "ở", "tại"...).
    * **Hành động:** Dùng 80 `candidate_ids` (TPHCM) được gửi.
    * Lọc chuyên sâu (Lớp 1+2) tìm `final_query = "bún bò"`.
    * Trả về các quán bún bò *ở TPHCM* có `S_taste` > 0.1.

#### Kịch bản 2: Lệnh Vị trí (Bỏ qua GPS User)

* **User gõ:** `"tôi muốn ăn phở ở hà nội"`
* **NestJS (Backend):** Lọc 20km (dựa trên GPS User ở TPHCM) -> Tìm được 80 quán `candidate_ids` (ID 1-80).
* **`api_fast.py` (AI):**
    * **Phát hiện "Lệnh Vị trí":** Lớp 1C (Tri Thức) "bắt" được cụm `" ở hà nội"`.
    * **Hành động:** Nó **BỎ QUA (IGNORE)** 80 `candidate_ids` (TPHCM) mà NestJS gửi.
    * **Lọc Mới (Internal):** Nó tự lọc *toàn bộ* `df` (155 quán) với `df['district'] == "Hà Nội"`.
    * **Lọc chuyên sâu:** Nó tìm `final_query = "phở"` trong 75 quán Hà Nội đó.
    * Trả về các quán phở *ở Hà Nội* có `S_taste` > 0.1.

#### Kịch bản 3: Lệnh Sở thích (Dùng GPS User)

* **User gõ:** `"tôi muốn ăn đặc sản hà nội"`
* **NestJS (Backend):** Lọc 20km (dựa trên GPS User ở TPHCM) -> Tìm được 80 quán `candidate_ids` (ID 1-80).
* **`api_fast.py` (AI):**
    * **Không phát hiện "Lệnh Vị trí"** (vì không có "ở", "tại"...).
    * **Hành động:** Nó **DÙNG** 80 `candidate_ids` (TPHCM) mà NestJS gửi.
    * **Dịch Query:** Lớp 1A (Tri Thức) "dịch" `"đặc sản hà nội"` -> `"phở bún chả bắc"`.
    * **Lọc chuyên sâu:** Nó tìm `final_query = "phở bún chả bắc"` trong 80 quán TPHCM đó.
    * Trả về các quán (ví dụ: "Phở Tàu Bay", "Bún Chả Huỳnh Thúc K") *ở TPHCM* có `S_taste` > 0.1.

#### Kịch bản 4: Lệnh Sắp xếp (Dùng GPS User)

* **User gõ:** `"tôi muốn ăn bún bò rẻ"`
* **NestJS (Backend):** Lọc 20km (dựa trên GPS User ở TPHCM) -> Tìm được 80 quán `candidate_ids` (ID 1-80).
* **`api_fast.py` (AI):**
    * **Không phát hiện "Lệnh Vị trí"**.
    * **Phát hiện "Lệnh Sắp xếp":** Lớp 1B (Tri Thức) "bắt" được cụm `" rẻ "` -> `sort_by = "price"`.
    * **Hành động:** Dùng 80 `candidate_ids` (TPHCM) được gửi.
    * **Lọc chuyên sâu:** Lớp 1+2 tìm `final_query = "bún bò"`.
    * **Trả về:** Server trả về 2 thông tin:
        1.  Danh sách các quán bún bò *ở TPHCM* có `S_taste` > 0.1.
        2.  Lệnh sắp xếp `sort_by = "price"`.
    * (NestJS sẽ nhận `sort_by="price"` và tự thực hiện logic xếp hạng này).