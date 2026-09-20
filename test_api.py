# -*- coding: utf-8 -*-
"""Offline end-to-end smoke test for the AI service.

Runs the whole FastAPI app against an injected fake collection, so it needs
no MongoDB and no network. Run it after any change to the AI service:

    cd AI && python test_api.py

The sentiment and YOLO models are loaded for real, so the first run
downloads weights and is slow.
"""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import data_store

FAKE = [
    {"_id": "a1", "tenQuan": "Phở Thìn Lò Đúc", "tags": ["phở", "phở bò", "sáng"],
     "diemTrungBinh": 8.6, "diaChi": "13 Lò Đúc, Hai Bà Trưng, Hà Nội",
     "giaCa": "50.000 - 90.000đ", "lat": 21.017, "lon": 105.856,
     "gioMoCua": "06:00 - 14:00", "avatarUrl": "", "urlGoc": "u1"},
    {"_id": "a2", "tenQuan": "Bún Bò Huế Đông Ba", "tags": ["bún bò huế", "bún bò", "cay"],
     "diemTrungBinh": 9.1, "diaChi": "110 Nguyễn Trãi, Quận 1, TPHCM",
     "giaCa": "60.000 - 100.000đ", "lat": 10.770, "lon": 106.695,
     "gioMoCua": "06:00 - 22:00", "avatarUrl": "", "urlGoc": "u2"},
    {"_id": "a3", "tenQuan": "Cơm Tấm Ba Ghiền", "tags": ["cơm tấm", "cơm sườn", "trưa"],
     "diemTrungBinh": 8.9, "diaChi": "84 Đặng Văn Ngữ, Phú Nhuận, TPHCM",
     "giaCa": "45.000 - 75.000đ", "lat": 10.795, "lon": 106.673,
     "gioMoCua": "08:00 - 21:00", "avatarUrl": "", "urlGoc": "u3"},
    {"_id": "a4", "tenQuan": "Ốc Đêm Quận 4", "tags": ["ốc", "hải sản", "ăn đêm", "đêm"],
     "diemTrungBinh": 7.8, "diaChi": "200 Vĩnh Khánh, Quận 4, TPHCM",
     "giaCa": "150.000 - 300.000đ", "lat": 10.759, "lon": 106.701,
     "gioMoCua": "17:00 - 02:00", "avatarUrl": "", "urlGoc": "u4"},
    {"_id": "a5", "tenQuan": "Cà Phê Trứng Giảng", "tags": ["cà phê trứng", "cà phê", "yên tĩnh"],
     "diemTrungBinh": 8.2, "diaChi": "39 Nguyễn Hữu Huân, Hoàn Kiếm, Hà Nội",
     "giaCa": "25.000 - 45.000đ", "lat": 21.033, "lon": 105.853,
     "gioMoCua": "07:00 - 22:00", "avatarUrl": "", "urlGoc": "u5"},
    {"_id": "a6", "tenQuan": "Bánh Mì Huỳnh Hoa", "tags": ["bánh mì", "nhanh"],
     "diemTrungBinh": 8.8, "diaChi": "26 Lê Thị Riêng, Quận 1, TPHCM",
     "giaCa": "60.000 - 70.000đ", "lat": 10.771, "lon": 106.692,
     "gioMoCua": "06:00 - 23:00", "avatarUrl": "", "urlGoc": "u6"},
]

# Inject the fake collection instead of reading Mongo.
data_store.RestaurantStore._fetch = lambda self: (self._normalize(FAKE), None)

from fastapi.testclient import TestClient
import api

def show(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

with TestClient(api.app) as client:
    show("GET /health")
    h = client.get("/health").json()
    print(" status:", h["status"], "| restaurants:", h["data"]["restaurants"])
    print(" models:", {k: v["ready"] for k, v in h["models"].items()})

    show("POST /recommend")
    for q, gps in [
        ("bún bò huế ngon ở quận 1", None),
        ("quán ăn đêm ở TPHCM", None),
        ("cà phê yên tĩnh Hà Nội", None),
        ("bánh mì rẻ", None),
        ("Phở Thìn Lò Đúc", None),
        ("đồ ăn gần đây", [10.770, 106.698]),
        ("hải sản dưới 50k", None),
        ("xyzzy không tồn tại", None),
    ]:
        r = client.post("/recommend", json={"query": q, "user_gps": gps}).json()
        names = [s["name"] for s in r["scores"][:3]]
        print(f"  {q!r:38} sort={r['sort_by']:10} n={len(r['scores'])}")
        print(f"      -> {names}")
        if r["relaxed_filters"]:
            print(f"      relaxed: {r['relaxed_filters']}")

    show("POST /chat  (multi-turn)")
    hist = []
    for msg in ["chào bạn", "bạn giúp được gì", "tôi đói", "phở ở Hà Nội", "rẻ hơn đi", "còn gì khác", "cảm ơn nha"]:
        r = client.post("/chat", json={"message": msg, "history": hist, "lang": "vi"}).json()
        print(f"\n  USER: {msg}")
        print(f"  BOT [{r['kind']}]: {r['reply'][:180]}")
        if r["results"]:
            print(f"        cards: {[x['name'] for x in r['results']]}")
        hist.append({"role": "user", "text": msg})
        hist.append({"role": "bot", "text": r["reply"]})

    show("POST /sentiment  +  /sentiment/batch")
    print(" single:", client.post("/sentiment", json={"review": "Đồ ăn ngon tuyệt vời!"}).json())
    b = client.post("/sentiment/batch", json={"reviews": [
        "Ngon lắm, sẽ quay lại", "Dở tệ, phí tiền", "Bình thường thôi"]}).json()
    print(" batch:", [(x["label"], round(x["score"], 3)) for x in b["results"]])

    show("POST /review-insights")
    r = client.post("/review-insights", json={"reviews": [
        {"noiDung": "Đồ ăn ngon, nước dùng đậm đà. Nhưng phục vụ chậm.", "diemReview": 8},
        {"noiDung": "Giá hợp lý, nhân viên thân thiện lắm.", "diemReview": 9},
        {"noiDung": "Quán bẩn, có ruồi. Nhân viên thái độ cọc.", "diemReview": 2},
        {"noiDung": "Món ăn ngon, giá rẻ. Không gian rộng thoáng.", "diemReview": 9},
    ], "lang": "vi"}).json()
    print(" summary:", r["summary"])
    for a in r["aspects"]:
        print(f"   {a['icon']} {a['label']}: {a['verdict']} ({a['positive']}+/{a['negative']}-)")

    show("Validation & auth")
    print(" bad gps ->", client.post("/recommend", json={"query": "phở", "user_gps": [999, 1]}).status_code, "(want 422)")
    print(" admin no token ->", client.post("/admin/reload").status_code, "(want 401/503)")
    print(" predict-food no file ->", client.post("/predict-food").status_code, "(want 422)")
    print(" parse-intent ->", json.dumps(
        {k: v for k, v in client.post("/parse-intent", json={"query": "lẩu dưới 200k ở quận 3 không cay"}).json().items()
         if v not in ([], None, False, "", "relevance")}, ensure_ascii=False))
