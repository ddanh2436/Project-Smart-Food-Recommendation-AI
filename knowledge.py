# -*- coding: utf-8 -*-
"""Vietnamese food and location knowledge base.

Four separate concerns, deliberately kept apart because the old single
``LOCATION_NAMES`` dict conflated them and produced false matches:

* ``EN_VI_MAPPING``    - English -> Vietnamese query translation.
* ``TAG_SYNONYMS``     - colloquial phrasing -> canonical tag.
* ``LOCATION_ALIASES`` - what a *user types* -> canonical place name.
* ``ADDRESS_PATTERNS`` - canonical place name -> regex matched against a
  restaurant's stored address.

Short abbreviations ("st", "bt", "tp") appear only in ``LOCATION_ALIASES``.
The old code matched them as bare substrings of addresses, so "st" hit inside
"Street" and "bt" was defined twice (Binh Thanh then Binh Thuy, the second
silently winning). Address scanning now uses full names with word boundaries.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------
# English -> Vietnamese. Deduplicated: the original listed 'pho', 'bbq' and
# 'luxury' twice each, so the later value silently overwrote the earlier one.
# --------------------------------------------------------------------------
EN_VI_MAPPING: dict[str, str] = {
    # -- Noodle soups --
    "beef noodle soup": "bún bò",
    "beef noodle": "bún bò",
    "bun bo hue": "bún bò huế",
    "bun bo": "bún bò",
    "noodle soup": "phở",
    "chicken noodle": "phở gà",
    "crab noodle": "bún riêu",
    "snail noodle": "bún ốc",
    "fish noodle": "bún cá",
    "bun dau mam tom": "bún đậu mắm tôm",
    "hu tieu": "hủ tiếu",
    "vermicelli": "bún",
    "glass noodle": "miến",
    "pho": "phở",
    "ramen": "mì nhật",
    "udon": "mì udon",
    # -- Rice & mains --
    "broken rice": "cơm tấm",
    "com tam": "cơm tấm",
    "pork chop rice": "cơm sườn",
    "chicken rice": "cơm gà",
    "fried rice": "cơm chiên",
    "sticky rice": "xôi",
    "rice": "cơm",
    "braised pork": "thịt kho",
    "catfish": "cá kho",
    # -- Bread & snacks --
    "banh mi": "bánh mì",
    "banhmi": "bánh mì",
    "banh-mi": "bánh mì",
    "baguette": "bánh mì",
    "sandwich": "bánh mì",
    "bread": "bánh mì",
    "sizzling cake": "bánh xèo",
    "pancake": "bánh xèo",
    "spring roll": "gỏi cuốn",
    "summer roll": "gỏi cuốn",
    "fresh roll": "gỏi cuốn",
    "goi-cuon": "gỏi cuốn",
    "goi cuon": "gỏi cuốn",
    "fried roll": "chả giò",
    "egg roll": "chả giò",
    "steamed roll": "bánh cuốn",
    "bot chien": "bột chiên",
    "dumpling": "há cảo",
    "dimsum": "dimsum",
    "street food": "vỉa hè",
    "snack": "ăn vặt",
    # -- Hotpot & grill --
    "thai hotpot": "lẩu thái",
    "hotpot": "lẩu",
    "grilled": "nướng",
    "bbq": "nướng",
    "beefsteak": "bít tết",
    "steak": "bít tết",
    # -- Ingredients --
    "seafood": "hải sản",
    "fish": "cá",
    "crab": "cua",
    "shrimp": "tôm",
    "snail": "ốc",
    "clam": "nghêu",
    "oyster": "hàu",
    "beef": "bò",
    "chicken": "gà",
    "pork": "heo",
    "duck": "vịt",
    "goat": "dê",
    "vegetarian": "chay",
    "vegan": "chay",
    "tofu": "đậu hũ",
    # -- Drinks & dessert --
    "egg coffee": "cà phê trứng",
    "milk coffee": "cà phê sữa",
    "coffee": "cà phê",
    "bubble tea": "trà sữa",
    "milk tea": "trà sữa",
    "tea": "trà",
    "juice": "nước ép",
    "smoothie": "sinh tố",
    "beer": "bia",
    "sweet soup": "chè",
    "ice cream": "kem",
    "dessert": "tráng miệng",
    "cake": "bánh ngọt",
    # -- Qualities --
    "delicious": "ngon",
    "yummy": "ngon",
    "tasty": "ngon",
    "best": "ngon nhất",
    "good": "ngon",
    "affordable": "rẻ",
    "reasonable": "rẻ",
    "budget": "rẻ",
    "cheap": "rẻ",
    "fine dining": "sang trọng",
    "luxury": "sang trọng",
    "expensive": "sang trọng",
    "closest": "gần",
    "nearby": "gần",
    "near": "gần",
    "spicy": "cay",
    "nice view": "view đẹp",
    "air conditioner": "máy lạnh",
    "quiet": "yên tĩnh",
    "family": "gia đình",
    "date night": "hẹn hò",
    "takeaway": "mang đi",
    # -- Time & place --
    "late night": "ăn đêm",
    "midnight": "đêm",
    "night": "đêm",
    "breakfast": "sáng",
    "morning": "sáng",
    "lunch": "trưa",
    "noon": "trưa",
    "dinner": "tối",
    "district": "quận",
    "hcmc": "tphcm",
    "saigon": "sài gòn",
    "city": "thành phố",
}

# --------------------------------------------------------------------------
# Colloquial phrasing -> canonical tag.
# --------------------------------------------------------------------------
TAG_SYNONYMS: dict[str, str] = {
    # Ambience / intent
    "lãng mạn": "hẹn hò",
    "sang chảnh": "sang trọng",
    "đắt tiền": "sang trọng",
    "thoải mái": "yên tĩnh",
    "nhanh gọn": "nhanh",
    "tụ tập": "nhậu",
    "nhậu nhẹt": "nhậu",
    "hạt dẻ": "rẻ",
    "sinh viên": "rẻ",
    "mát mẻ": "máy lạnh",
    "điều hòa": "máy lạnh",
    "sống ảo": "view đẹp",
    "chill": "yên tĩnh",
    # Regional dishes
    "đồng quê": "cơm việt",
    "cơm bắc": "cơm việt",
    "cơm niêu": "cơm việt",
    "đặc sản huế": "bún bò huế",
    "món huế": "bún bò huế",
    "đặc sản hà nội": "bún chả",
    "phở bắc": "phở",
    "đặc sản sài gòn": "cơm tấm",
    "đồ nướng": "nướng",
    "hải sản tươi sống": "hải sản",
}

# --------------------------------------------------------------------------
# Canonical place names, grouped by the city they belong to.
# --------------------------------------------------------------------------
CITY_DISTRICTS: dict[str, list[str]] = {
    "TPHCM": [
        "Quận 1", "Quận 2", "Quận 3", "Quận 4", "Quận 5", "Quận 6",
        "Quận 7", "Quận 8", "Quận 9", "Quận 10", "Quận 11", "Quận 12",
        "Bình Thạnh", "Phú Nhuận", "Gò Vấp", "Tân Bình", "Tân Phú",
        "Bình Tân", "Thủ Đức", "Bình Chánh", "Hóc Môn", "Củ Chi",
        "Nhà Bè", "Cần Giờ",
    ],
    "Hà Nội": [
        "Ba Đình", "Hoàn Kiếm", "Tây Hồ", "Long Biên", "Cầu Giấy",
        "Đống Đa", "Hai Bà Trưng", "Hoàng Mai", "Thanh Xuân", "Sóc Sơn",
        "Đông Anh", "Gia Lâm", "Nam Từ Liêm", "Bắc Từ Liêm", "Hà Đông",
        "Sơn Tây", "Ba Vì", "Phúc Thọ", "Đan Phượng", "Hoài Đức",
        "Quốc Oai", "Thạch Thất", "Chương Mỹ", "Thanh Oai", "Thường Tín",
        "Phú Xuyên", "Ứng Hòa", "Mỹ Đức", "Mê Linh", "Thanh Trì",
    ],
    "Đà Nẵng": [
        "Hải Châu", "Thanh Khê", "Sơn Trà", "Ngũ Hành Sơn",
        "Liên Chiểu", "Cẩm Lệ", "Hòa Vang",
    ],
    "Cần Thơ": [
        "Ninh Kiều", "Cái Răng", "Bình Thủy", "Ô Môn", "Thốt Nốt",
        "Phong Điền",
    ],
    "Hải Phòng": [
        "Ngô Quyền", "Lê Chân", "Hồng Bàng", "Hải An", "Kiến An",
        "Đồ Sơn", "Dương Kinh", "Thủy Nguyên", "An Dương", "An Lão",
        "Cát Hải", "Cát Bà",
    ],
    "Khánh Hòa": [
        "Nha Trang", "Cam Ranh", "Ninh Hòa", "Vạn Ninh",
        "Diên Khánh", "Cam Lâm",
    ],
    "Vũng Tàu": [
        "Bà Rịa", "Phú Mỹ", "Châu Đức", "Xuyên Mộc", "Long Điền",
        "Đất Đỏ", "Côn Đảo",
    ],
    "Lâm Đồng": [
        "Đà Lạt", "Bảo Lộc", "Đức Trọng", "Di Linh", "Lạc Dương",
        "Đơn Dương", "Lâm Hà",
    ],
}

MAJOR_CITIES: list[str] = list(CITY_DISTRICTS.keys())

DISTRICT_TO_CITY: dict[str, str] = {
    district: city
    for city, districts in CITY_DISTRICTS.items()
    for district in districts
}

# --------------------------------------------------------------------------
# What a user may type -> canonical place. Matched with word boundaries, so
# short forms are safe here (they are never used against stored addresses).
# --------------------------------------------------------------------------
LOCATION_ALIASES: dict[str, str] = {
    # Cities and nicknames
    "sài gòn": "TPHCM", "saigon": "TPHCM", "sai gon": "TPHCM",
    "hcm": "TPHCM", "tphcm": "TPHCM", "tp hcm": "TPHCM",
    "hồ chí minh": "TPHCM", "ho chi minh": "TPHCM",
    "hà nội": "Hà Nội", "ha noi": "Hà Nội", "hanoi": "Hà Nội",
    "hn": "Hà Nội", "thủ đô": "Hà Nội",
    "đà nẵng": "Đà Nẵng", "da nang": "Đà Nẵng", "đn": "Đà Nẵng",
    "cần thơ": "Cần Thơ", "can tho": "Cần Thơ", "tây đô": "Cần Thơ",
    "hải phòng": "Hải Phòng", "hai phong": "Hải Phòng",
    "đất cảng": "Hải Phòng", "hp": "Hải Phòng",
    "khánh hòa": "Khánh Hòa",
    "nha trang": "Nha Trang", "nhatrang": "Nha Trang",
    "vũng tàu": "Vũng Tàu", "vung tau": "Vũng Tàu", "brvt": "Vũng Tàu",
    "bà rịa vũng tàu": "Vũng Tàu",
    "lâm đồng": "Lâm Đồng",
    "đà lạt": "Đà Lạt", "da lat": "Đà Lạt", "dalat": "Đà Lạt",
    "thành phố ngàn hoa": "Đà Lạt", "xứ sở sương mù": "Đà Lạt",
    # HCMC numbered districts (longest-first matching handles 1 vs 12)
    **{f"quận {n}": f"Quận {n}" for n in range(1, 13)},
    **{f"q{n}": f"Quận {n}" for n in range(1, 13)},
    **{f"q.{n}": f"Quận {n}" for n in range(1, 13)},
    **{f"district {n}": f"Quận {n}" for n in range(1, 13)},
    # HCMC named districts
    "bình thạnh": "Bình Thạnh", "binh thanh": "Bình Thạnh",
    "phú nhuận": "Phú Nhuận", "phu nhuan": "Phú Nhuận",
    "gò vấp": "Gò Vấp", "go vap": "Gò Vấp",
    "tân bình": "Tân Bình", "tan binh": "Tân Bình",
    "tân phú": "Tân Phú", "tan phu": "Tân Phú",
    "bình tân": "Bình Tân", "binh tan": "Bình Tân",
    "thủ đức": "Thủ Đức", "thu duc": "Thủ Đức",
    "bình chánh": "Bình Chánh", "hóc môn": "Hóc Môn",
    "củ chi": "Củ Chi", "nhà bè": "Nhà Bè", "cần giờ": "Cần Giờ",
    # Hanoi districts
    "ba đình": "Ba Đình", "ba dinh": "Ba Đình",
    "hoàn kiếm": "Hoàn Kiếm", "hoan kiem": "Hoàn Kiếm",
    "tây hồ": "Tây Hồ", "tay ho": "Tây Hồ",
    "long biên": "Long Biên", "long bien": "Long Biên",
    "cầu giấy": "Cầu Giấy", "cau giay": "Cầu Giấy",
    "đống đa": "Đống Đa", "dong da": "Đống Đa",
    "hai bà trưng": "Hai Bà Trưng", "hbt": "Hai Bà Trưng",
    "hoàng mai": "Hoàng Mai", "thanh xuân": "Thanh Xuân",
    "sóc sơn": "Sóc Sơn", "đông anh": "Đông Anh", "gia lâm": "Gia Lâm",
    "nam từ liêm": "Nam Từ Liêm", "bắc từ liêm": "Bắc Từ Liêm",
    "hà đông": "Hà Đông", "sơn tây": "Sơn Tây", "thanh trì": "Thanh Trì",
    # Da Nang
    "hải châu": "Hải Châu", "thanh khê": "Thanh Khê",
    "sơn trà": "Sơn Trà", "ngũ hành sơn": "Ngũ Hành Sơn",
    "liên chiểu": "Liên Chiểu", "cẩm lệ": "Cẩm Lệ", "hòa vang": "Hòa Vang",
    # Can Tho / Hai Phong / others
    "ninh kiều": "Ninh Kiều", "cái răng": "Cái Răng",
    "bình thủy": "Bình Thủy", "ô môn": "Ô Môn", "thốt nốt": "Thốt Nốt",
    "ngô quyền": "Ngô Quyền", "lê chân": "Lê Chân", "hồng bàng": "Hồng Bàng",
    "đồ sơn": "Đồ Sơn", "cát bà": "Cát Bà",
    "cam ranh": "Cam Ranh", "bảo lộc": "Bảo Lộc",
}

# --------------------------------------------------------------------------
# Address matching. Built from full names only (plus an accent-free variant),
# so scanning an address can never trip over a 2-letter abbreviation.
# --------------------------------------------------------------------------
_ACCENT_MAP = str.maketrans(
    "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
    "ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ",
    "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd"
    "AAAAAAAAAAAAAAAAAEEEEEEEEEEEIIIIIOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYD",
)


def strip_accents(text: str) -> str:
    """Drop Vietnamese diacritics so 'Ha Noi' also matches 'Hà Nội'."""
    return text.translate(_ACCENT_MAP)


def _numbered_variants(canonical: str) -> set[str]:
    """Extra spellings for 'Quận N' as it appears in real addresses."""
    match = re.fullmatch(r"Quận (\d+)", canonical)
    if not match:
        return set()
    n = match.group(1)
    return {
        f"Quan {n}", f"Q.{n}", f"Q. {n}", f"Q {n}", f"Q{n}", f"District {n}",
    }


def _boundary_regex(names: set[str]) -> re.Pattern[str]:
    """Word-boundary alternation over ``names``, longest alternative first.

    ``\\w`` lookarounds rather than ``\\b`` because ``\\b`` behaves badly next
    to the '.' in "Q.1" and next to accented characters.
    """
    alternatives = "|".join(
        re.escape(name) for name in sorted(names, key=len, reverse=True)
    )
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)


ADDRESS_PATTERNS: dict[str, re.Pattern[str]] = {
    name: _boundary_regex(
        {name, strip_accents(name)} | _numbered_variants(name)
    )
    for name in list(DISTRICT_TO_CITY) + MAJOR_CITIES
}

# A city-level pattern also accepts any of its districts, because an address in
# "Ba Đình" is in Hà Nội even when the city itself is never written out.
CITY_PATTERNS: dict[str, re.Pattern[str]] = {}
for _city, _districts in CITY_DISTRICTS.items():
    _names = {_city, strip_accents(_city)}
    if _city == "TPHCM":
        _names |= {"Hồ Chí Minh", "Ho Chi Minh", "TP.HCM", "TP HCM", "Sài Gòn",
                   "Sai Gon", "HCM"}
    for _district in _districts:
        _names |= {_district, strip_accents(_district)}
        _names |= _numbered_variants(_district)
    CITY_PATTERNS[_city] = _boundary_regex(_names)

# The frontend sends these short keys; map them to canonical places.
CITY_KEY_ALIASES: dict[str, str] = {
    "hanoi": "Hà Nội",
    "hcmc": "TPHCM",
    "danang": "Đà Nẵng",
    "cantho": "Cần Thơ",
    "haiphong": "Hải Phòng",
    "nhatrang": "Nha Trang",
    "vungtau": "Vũng Tàu",
    "dalat": "Đà Lạt",
}

# --------------------------------------------------------------------------
# Tag vocabulary.
# --------------------------------------------------------------------------
CANDIDATE_TAGS: list[str] = [
    # Noodle soups
    "bún bò huế", "bún bò", "bún riêu", "bún mắm", "bún chả",
    "bún thịt nướng", "bún đậu mắm tôm", "bún đậu", "bún cá", "bún mọc",
    "bún thái", "bún ốc", "bún",
    "phở bò", "phở gà", "phở cuốn", "phở",
    "hủ tiếu nam vang", "hủ tiếu gõ", "hủ tiếu mực", "hủ tiếu",
    "bánh canh cua", "bánh canh ghẹ", "bánh canh cá lóc", "bánh canh",
    "mì quảng", "mì vịt tiềm", "mì ý", "mì cay", "mì xào", "mì trộn",
    "mì nhật", "mì udon", "mì",
    "ramen", "udon", "miến gà", "miến lươn", "miến xào", "miến",
    "nui xào", "nui", "bò kho", "lagu", "cà ri",
    "cháo lòng", "cháo ếch", "cháo", "súp",
    # Rice
    "cơm tấm", "cơm sườn", "cơm gà xối mỡ", "cơm gà", "cơm niêu",
    "cơm văn phòng", "cơm chiên", "cơm rang", "cơm lam", "cơm phần",
    "cơm việt", "cơm", "xôi gà", "xôi mặn", "xôi",
    # Mains / drinking food
    "lẩu thái", "lẩu bò", "lẩu gà", "lẩu dê", "lẩu hải sản", "lẩu mắm",
    "lẩu cá", "lẩu",
    "bò nướng", "gà nướng", "hải sản nướng", "nem nướng", "nướng", "bbq",
    "bò bít tết", "bít tết", "bò né", "steak",
    "hải sản", "ốc", "tôm", "cua", "ghẹ", "hàu", "mực", "bạch tuộc",
    "gà rán", "gà luộc", "gà ủ muối", "vịt quay", "heo quay", "phá lấu",
    "dê", "cừu", "ếch", "lươn", "bột chiên", "thịt kho", "cá kho",
    # Bread & snacks
    "bánh mì chảo", "bánh mì xíu mại", "bánh mì",
    "bánh xèo", "bánh khọt", "bánh cuốn", "bánh ướt", "bánh bèo",
    "bánh bột lọc", "bánh nậm",
    "gỏi cuốn", "bì cuốn", "chả giò", "nem rán",
    "pizza", "hamburger", "sushi", "sashimi", "dimsum", "há cảo", "xíu mại",
    "bánh tráng trộn", "cá viên chiên", "xiên que", "bắp xào",
    "hột vịt lộn", "ăn vặt",
    # Drinks & dessert
    "cà phê trứng", "cà phê sữa", "cà phê vợt", "cà phê",
    "trà sữa", "trà đào", "trà chanh", "trà",
    "sinh tố", "nước ép", "chè", "kem", "bingsu", "tàu hũ", "sữa chua",
    "bánh ngọt", "tráng miệng",
    "bia thủ công", "bia", "rượu", "pub", "bar",
    # Style / cuisine
    "thuần chay", "chay", "healthy", "eat clean", "đậu hũ",
    "hàn quốc", "nhật bản", "trung hoa", "thái lan", "âu", "mỹ", "ý",
    "vỉa hè", "sang trọng", "bình dân", "gia đình", "hẹn hò", "nhậu",
    "view đẹp", "máy lạnh", "sân vườn", "yên tĩnh", "nhanh", "mang đi",
    "buffet", "cay",
    # Time
    "ăn đêm", "24h", "sáng", "trưa", "chiều", "tối", "đêm",
]

# Tags that describe *when* a place is useful. Applied as hard filters.
TIME_TAGS: list[str] = ["đêm", "sáng", "trưa", "chiều", "tối", "ăn đêm", "24h"]

# Tags that describe a preference rather than a dish. Never used to filter
# the candidate set, only to rank it.
ADJECTIVE_TAGS: list[str] = [
    "rẻ", "gần", "ngon", "tốt", "nhanh", "đẹp", "vỉa hè", "sang trọng",
    "yên tĩnh", "nổi tiếng", "nhất", "bình dân", "gia đình", "hẹn hò",
    "máy lạnh", "sân vườn", "mang đi", "view đẹp", "healthy", "eat clean",
    "buffet", "nhậu",
]

# Longest-first so "bún bò huế" beats "bún bò" and "Quận 12" beats "Quận 1".
CANDIDATE_TAGS_SORTED: list[str] = sorted(CANDIDATE_TAGS, key=len, reverse=True)
LOCATION_ALIASES_SORTED: list[str] = sorted(LOCATION_ALIASES, key=len, reverse=True)
EN_VI_SORTED: list[str] = sorted(EN_VI_MAPPING, key=len, reverse=True)
TAG_SYNONYMS_SORTED: list[str] = sorted(TAG_SYNONYMS, key=len, reverse=True)

# Dish tags only: everything that is not purely an adjective or a time hint.
DISH_TAGS: set[str] = {
    tag for tag in CANDIDATE_TAGS
    if tag not in ADJECTIVE_TAGS and tag not in TIME_TAGS
}


# --------------------------------------------------------------------------
# Concepts: everyday phrasing that stands for a set of slots.
#
# "món nước kiểu miền trung ở khu trung tâm, giá sinh viên" names no dish, no
# district and no price, yet it says all three. The rules above only know
# literal tags, so the whole sentence fell through to free text -- and, being
# several unknown words, was then taken for a restaurant's name. A concept
# maps such a phrase onto the slots the search already understands:
#
#   dishes     -> a filter; any one of them matches ("món nước" = phở OR bún…)
#   adjectives -> ranking only, matched against the restaurant's tags
#   aspects    -> ranking on the review verdicts (see intent.ASPECT_CUES)
#   time_tags, districts, price_max, sort_by -> as in the Intent
#
# Every value here is a tag, district or aspect that exists in the data, so a
# concept can widen a search but never invent a place. Cues are matched
# before colloquial synonyms (TAG_SYNONYMS), longest first, and consumed.
# --------------------------------------------------------------------------
CONCEPTS: list[dict] = [
    # -- Kinds of dish ----------------------------------------------------
    {
        "name": "soup_dishes",
        "cues": ["món có nước", "món nước", "đồ nước", "món súp", "món chan nước"],
        # Not bare "mì": tags are matched as substrings, and "mì" is inside
        # "bánh mì", which is about as far from soup as a dish gets.
        "dishes": [
            "phở", "bún", "hủ tiếu", "bánh canh", "mì quảng", "mì vịt tiềm",
            "miến", "cháo",
        ],
    },
    {
        "name": "dry_dishes",
        "cues": ["món khô", "đồ khô"],
        "dishes": ["cơm", "bánh mì", "xôi", "bún thịt nướng", "mì trộn", "bánh xèo"],
    },
    {
        "name": "warming_food",
        "cues": [
            "trời lạnh", "trời mưa", "ấm bụng", "nóng hổi", "món nóng",
            "nóng nóng", "đồ nóng", "ấm người", "cho ấm",
        ],
        "dishes": ["lẩu", "phở", "bún", "cháo", "hủ tiếu", "bánh canh"],
    },
    {
        "name": "cooling_food",
        "cues": [
            "trời nóng", "giải nhiệt", "mát lạnh", "giải khát", "đồ mát",
            "món mát", "uống gì mát",
        ],
        "dishes": ["chè", "kem", "sinh tố", "nước ép", "trà", "bingsu"],
    },
    {
        "name": "light_bite",
        "cues": ["ăn nhẹ", "ăn chơi", "lót dạ", "ăn xế", "đồ ăn vặt"],
        "dishes": ["ăn vặt", "bánh tráng trộn", "xôi", "bánh mì", "chè"],
    },
    {
        "name": "filling_food",
        "cues": ["ăn no", "no bụng", "no nê", "ăn cho no", "cho no"],
        "dishes": ["cơm", "lẩu", "bún", "phở"],
    },
    {
        "name": "seafood",
        "cues": ["đồ biển", "món biển", "hải sản tươi"],
        "dishes": ["hải sản", "ốc"],
    },
    {
        "name": "office_lunch",
        "cues": [
            "cơm trưa văn phòng", "ăn trưa văn phòng", "dân văn phòng",
            "cơm văn phòng", "đồ ăn trưa",
        ],
        "dishes": ["cơm văn phòng", "cơm"],
    },
    {
        "name": "healthy",
        "cues": [
            # Not "ăn sạch": it is the start of "quán ăn sạch sẽ", which asks
            # for hygiene, not for a diet.
            "ăn kiêng", "giảm cân", "ít dầu mỡ", "lành mạnh", "thanh đạm",
            "nhẹ bụng",
        ],
        "adjectives": ["healthy", "eat clean", "chay"],
    },
    # -- Regional cooking: ranking, not a filter, because only some places
    #    carry the regional tag and a filter would drop the rest. ----------
    {
        "name": "central_cooking",
        "cues": [
            "kiểu miền trung", "món miền trung", "đồ miền trung", "miền trung",
            "món trung",
        ],
        "adjectives": ["món miền trung", "món huế"],
    },
    {
        "name": "northern_cooking",
        "cues": ["kiểu miền bắc", "món miền bắc", "đồ bắc", "miền bắc", "món bắc"],
        "adjectives": ["món bắc"],
    },
    {
        "name": "southern_cooking",
        "cues": ["kiểu miền nam", "món miền nam", "miền nam", "món nam bộ"],
        "adjectives": ["món miền nam"],
    },
    {
        "name": "mekong_cooking",
        "cues": ["kiểu miền tây", "món miền tây", "đồ miền tây", "miền tây"],
        "adjectives": ["món miền tây"],
    },
    {
        "name": "northwest_cooking",
        "cues": ["tây bắc", "vùng cao"],
        "adjectives": ["món tây bắc"],
    },
    # -- Budget -------------------------------------------------------------
    {
        "name": "homestyle",
        "cues": ["dân dã", "bình dị", "kiểu nhà làm", "cơm nhà", "cơm mẹ nấu"],
        "adjectives": ["bình dân", "cơm việt"],
    },
    {
        "name": "student_budget",
        "cues": [
            "giá sinh viên", "túi tiền sinh viên", "sinh viên", "hạt dẻ",
            "giá bèo", "rẻ bèo", "siêu rẻ",
        ],
        "adjectives": ["bình dân"],
        "price_max": 60_000,
        "sort_by": "price",
    },
    {
        "name": "moderate_budget",
        "cues": [
            "không quá đắt", "không đắt lắm", "không đắt", "đừng quá đắt",
            "vừa túi tiền", "giá vừa phải", "giá phải chăng", "tầm trung",
        ],
        "price_max": 150_000,
        "aspects": ["price"],
    },
    # -- Occasion and company ---------------------------------------------
    {
        "name": "date",
        "cues": [
            "đi hẹn hò", "hẹn hò", "đi date", "người yêu", "bạn gái",
            "bạn trai", "lãng mạn", "kỷ niệm ngày", "crush",
        ],
        "adjectives": ["hẹn hò"],
        "aspects": ["space"],
    },
    {
        "name": "quiet",
        "cues": [
            "không quá ồn", "đừng quá ồn", "không ồn ào", "không ồn", "đừng ồn",
            "ít ồn", "yên tĩnh", "riêng tư", "chill",
        ],
        "adjectives": ["yên tĩnh"],
        "aspects": ["space"],
    },
    {
        "name": "family",
        "cues": [
            "đi với con", "có con nhỏ", "cho trẻ em", "trẻ con", "cả nhà",
            "bố mẹ", "ông bà", "gia đình",
        ],
        "adjectives": ["gia đình"],
    },
    {
        "name": "group",
        "cues": [
            "đông người", "nhóm bạn", "liên hoan", "sinh nhật", "họp lớp",
            "tụ tập",
        ],
        "adjectives": ["tụ tập", "nhóm hội"],
    },
    {
        "name": "business",
        "cues": ["tiếp khách", "đối tác", "gặp khách", "mời sếp", "tiếp đối tác"],
        "adjectives": ["tiếp khách", "sang trọng"],
        "aspects": ["service"],
    },
    {
        "name": "drinks",
        "cues": [
            "lai rai", "nhâm nhi", "làm vài ly", "đi nhậu", "uống bia",
            "nhậu nhẹt",
        ],
        "adjectives": ["nhậu"],
    },
    {
        "name": "photogenic",
        "cues": ["check in", "checkin", "sống ảo", "chụp ảnh đẹp", "view đẹp"],
        "adjectives": ["view đẹp"],
        "aspects": ["space"],
    },
    # -- When ---------------------------------------------------------------
    {
        "name": "late_night",
        "cues": ["về khuya", "nửa đêm", "đêm muộn", "khuya"],
        "time_tags": ["ăn đêm"],
    },
    {
        "name": "breakfast",
        "cues": ["bữa sáng", "điểm tâm"],
        "time_tags": ["sáng"],
    },
    # -- Where --------------------------------------------------------------
    {
        "name": "saigon_centre",
        "cues": [
            "trung tâm sài gòn", "trung tâm thành phố hồ chí minh",
            "trung tâm tp hcm", "trung tâm tphcm", "trung tâm hcm",
        ],
        "districts": ["Quận 1", "Quận 3"],
    },
    {
        "name": "hanoi_centre",
        "cues": ["trung tâm hà nội", "phố cổ", "hồ gươm", "bờ hồ", "hồ hoàn kiếm"],
        "districts": ["Hoàn Kiếm"],
    },
    {
        "name": "city_centre",
        "cues": ["khu trung tâm", "quận trung tâm", "trung tâm"],
        "districts": ["Quận 1", "Quận 3", "Hoàn Kiếm", "Hải Châu"],
    },
]

# (cue, concept) pairs, longest cue first, so "không quá đắt" is read before
# "không đắt" and "trung tâm sài gòn" before "trung tâm".
CONCEPT_CUES_SORTED: list[tuple[str, dict]] = sorted(
    ((cue, concept) for concept in CONCEPTS for cue in concept["cues"]),
    key=lambda pair: len(pair[0]),
    reverse=True,
)
