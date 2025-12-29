import pandas as pd
import time as time_module
import numpy as np
import sys
import json
import os
import re
import io
import contextlib
import traceback
from dotenv import load_dotenv
from pulp import *
from time import sleep
from huggingface_hub import InferenceClient

# Đặt encoding utf-8 cho stdout ngay đầu file
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# tiny helper: log to stderr so stdout remains clean JSON
def log(*args, **kwargs):
    """
    Viết log ra stderr để stdout chỉ chứa JSON kết quả.
    Dùng thay cho print(...) để tránh pollute stdout.
    """
    print(*args, file=sys.stderr, **kwargs)

# ----------------------------------------------------
# CẤU HÌNH API LLM & MÔ HÌNH
# ----------------------------------------------------
# Sử dụng mô hình bạn đã xác nhận là hoạt động
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct" 

# 1. TÍNH TOÁN ĐƯỜNG DẪN GỐC DỰ ÁN (PROJECT_ROOT)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '..', '..')

# 2. THÊM THƯ MỤC GỐC DỰ ÁN VÀO sys.path
sys.path.append(PROJECT_ROOT)

# 3. IMPORT CÁC MODULE CON
from healthyApp.foodDB import FoodDatabase
from healthyApp.utils import load_data_csv, get_faiss_index, get_food_index

# Tải biến môi trường
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, '.env'))

# Tắt cảnh báo symlink
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'true'

try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

HF_TOKEN = os.getenv("HF_TOKEN") # optional, nếu có sẽ dùng để gọi HF Inference

# -----------------------
# PREPROCESS / DEDUPE / TRUNCATE
# -----------------------
def normalize_name(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\(.*?\)", "", s)             # loại bỏ nội dung trong ()
    s = re.sub(r"[^a-z0-9]+", " ", s)         # chỉ giữ ký tự alnum
    return " ".join(s.split())

def preprocess_results(raw_items, max_items=40, max_chars=15000, max_chars_per_item=400):
    log(f"Starting preprocess_results with {len(raw_items)} raw items, Timestamp: {time_module.strftime('%Y-%m-%d %H:%M:%S')}")
    """
    raw_items: list[dict] - mỗi dict chứa at least: 'fdcId', 'description', 'nutrients', 'ingredients'
    Trả về danh sách đã rút gọn, dedupe theo tên, giới hạn kích thước.
    """
    seen = set()
    clean = []
    total_chars = 0

    for item in raw_items:
        name = item.get("description") or item.get("name") or ""
        key = normalize_name(name)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)

        # rút gọn nutrients: giữ tối đa 3 nutrient chính nếu có
        nutrients = item.get("nutrients") or {}
        short_nutrients = {}
        if isinstance(nutrients, dict):
            try:
                sorted_items = sorted(nutrients.items(), key=lambda x: float(x[1]) if x[1] is not None else 0, reverse=True)
                for k, v in sorted_items[:3]:
                    short_nutrients[k] = v
            except Exception:
                short_nutrients = nutrients
        else:
            short_nutrients = nutrients

        # rút gọn ingredients nếu quá dài
        ingredients = item.get("ingredients") or []
        if isinstance(ingredients, list):
            if len(ingredients) > 20:
                ingredients = ingredients[:20]
        else:
            ingredients = []

        # truncate description
        desc = (item.get("description") or "").strip()
        if len(desc) > max_chars_per_item:
            desc = desc[:max_chars_per_item].rstrip() + " ..."

        clean_item = {
            "fdcId": int(item.get("fdcId")) if item.get("fdcId") is not None else None,
            "name": name.strip(),
            "description": desc,
            "nutrients": short_nutrients,
            "ingredients": ingredients
        }

        item_size = len(json.dumps(clean_item, ensure_ascii=False))
        if total_chars + item_size > max_chars:
            break
        total_chars += item_size

        clean.append(clean_item)
        if len(clean) >= max_items:
            break
    log(f"Preprocess completed, returning {len(clean)} items")
    return clean

# -----------------------
# BUILD PROMPT & SUMMARIZE
# -----------------------
def fallback_summary(prepped_items, query, max_items=6):
    items = prepped_items[:max_items]
    if not items:
        return f"Không tìm thấy món ăn nào cho \"{query}\". Vui lòng thử lại với từ khóa khác."
    names = [it["name"] for it in items]
    s = f"Tìm thấy {len(prepped_items)} món ăn cho \"{query}\". Một số lựa chọn: {', '.join(names[:5])}."
    if items and items[0].get("nutrients"):
        s += "\nVí dụ dinh dưỡng: "
        for it in items[:3]:
            top_n = ", ".join(list(it["nutrients"].keys())[:2])
            s += f"{it['name']} (chứa {top_n}); "
    s += "\nHãy cho tôi biết nếu bạn muốn kế hoạch ăn uống chi tiết!"
    return s

def build_summary_prompt(prepped_items, query, max_items_in_prompt=8, user_info=None):
    if not prepped_items:
        items = []
    else:
        items = prepped_items[:max_items_in_prompt]

    try:
        items_json = json.dumps(items, ensure_ascii=False, indent=2)
    except Exception:
        items_json = str(items)

    user_info_str = f"Thông tin người dùng: {json.dumps(user_info, ensure_ascii=False, indent=2)}\n\n" if user_info else ""
    
    instructions = (
        f"Bạn là một trợ lý ẩm thực và dinh dưỡng, chuyên gia về sức khỏe cá nhân. {user_info_str}"
        "Dưới đây là dữ liệu JSON gồm các thực phẩm/nguyên liệu (mỗi phần tử có fdcId, name, description, nutrients, ingredients).\n\n"
        "JSON_DANH_SACH:\n"
        f"{items_json}\n\n"
        "Yêu cầu:\n"
        "1) Viết một câu trả lời bằng TIẾNG VIỆT, thân thiện và dễ hiểu cho người dùng.\n"
        "2) Bắt đầu bằng một tóm tắt ngắn (1-2 câu) nêu tổng quan những lựa chọn phù hợp với truy vấn: "
        f"\"{query}\".\n"
        "3) Dựa trên DỮ LIỆU CÁ NHÂN (dị ứng, bệnh, mục tiêu) và DỮ LIỆU THỰC PHẨM (JSON) để đưa ra lời khuyên an toàn và cá nhân hóa.\n"
        "4) Gợi ý 2-4 cách sử dụng hoặc món ăn/kế hoạch ăn nhanh có thể làm từ các thực phẩm này.\n"
        "5) Kết thúc bằng một câu hỏi ngắn khuyến khích người dùng hỏi chi tiết hơn (ví dụ: công thức, kế hoạch 3 ngày, lời khuyên cụ thể cho bệnh lý).\n\n"
        "Lưu ý: KHÔNG lặp lại nguyên văn JSON trong phần trả lời. Ưu tiên độ an toàn và sự phù hợp dinh dưỡng.\n"
    )

    return instructions

def generate_summary_via_hf(prompt, model_name=None, max_tokens=2000, prepped_items=None):
    client = None 
    
    # KHÔNG CẦN DÙNG OS.ENVIRON.POP() nữa vì lỗi Provider đã được giải quyết bằng mô hình khác.
    
    try:
        # **FIX:** Buộc sử dụng provider='hf-inference' để tránh lỗi định tuyến 401
        client = InferenceClient(
            model=model_name or DEFAULT_HF_MODEL,
            provider='auto', 
            token=HF_TOKEN
        )
        log(f"Starting HF API call with prompt length: {len(prompt)} chars, Timestamp: {time_module.strftime('%Y-%m-%d %H:%M:%S')}")

        if client is None:
             raise RuntimeError("Hugging Face Client không thể khởi tạo.")
        
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            stream=True
        )

        summary = ''
        chunk_buffer = ''
        min_chunk_size = 100 # Mỗi chunk tối thiểu 100 ký tự

        # Gửi từng phần giống như ChatGPT
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                text_piece = chunk.choices[0].delta.content
                summary += text_piece
                chunk_buffer += text_piece

                # Khi buffer đủ dài hoặc kết thúc câu → gửi ra stdout (JSON line)
                if len(chunk_buffer) >= min_chunk_size or chunk_buffer.endswith(('.', '!', '?', '\n')):
                    sys.stdout.write(json.dumps({"type": "chunk", "content": chunk_buffer}) + "\n")
                    sys.stdout.flush()
                    chunk_buffer = ''

        # Gửi phần còn lại
        if chunk_buffer:
            sys.stdout.write(json.dumps({"type": "chunk", "content": chunk_buffer}) + "\n")
            sys.stdout.flush()

        # Gửi tín hiệu hoàn tất
        sys.stdout.write(json.dumps({"type": "complete", "summary": summary}) + "\n")
        sys.stdout.flush()

        log(f"Streaming completed successfully. Response length: {len(summary)} chars")
        return summary

    except Exception as e:
        log(f"Error during HF API call: {str(e)}")
        sys.stdout.write(json.dumps({"type": "error", "error": str(e)}) + "\n")
        sys.stdout.flush()
        return None
    # KHÔNG CẦN KHOI PHỤC BIẾN MÔI TRƯỜNG NỮA

# -----------------------
# PHÂN LOẠI Ý ĐỊNH (INTENT CLASSIFICATION)
# -----------------------
def classify_intent(query):
    """Phân loại truy vấn của người dùng thành các chủ đề chính."""
    query_lower = query.lower()
    
    # 1. Kế hoạch ăn uống/Chế độ (Độ ưu tiên cao nhất)
    keywords_plan = ["kế hoạch", "thực đơn", "chế độ", "ăn kiêng", "cân bằng", "trong ngày", "cho bữa"]
    if any(k in query_lower for k in keywords_plan):
        return "PLANNING"
        
    # 2. Sức khỏe/Tư vấn/Bệnh lý
    keywords_health = ["sức khỏe", "bệnh", "tư vấn", "ăn uống", "tăng cân", "giảm cân", "dị ứng", "huyết áp", "tiểu đường", "thận", "ung thư"]
    if any(k in query_lower for k in keywords_health):
        return "HEALTH_ADVICE"

    # 3. Dinh dưỡng/Thông tin thực phẩm cụ thể
    keywords_nutrient = ["dinh dưỡng", "calo", "protein", "chất béo", "có bao nhiêu", "vitamin", "khoáng chất", "chứa"]
    if any(k in query_lower for k in keywords_nutrient):
        return "FOOD_INFO"

    # 4. Truy vấn chung không liên quan đến dinh dưỡng
    if len(query.split()) < 3 and ("là gì" in query_lower or "ai là" in query_lower):
        return "GENERAL_OTHER" # Ví dụ: "Thủ đô Việt Nam là gì?"

    # Mặc định là tìm kiếm thực phẩm và đưa ra tóm tắt
    return "SEARCH_SUMMARY" 

# -----------------------
# RUN SEARCH (CHÍNH)
# -----------------------
def run_food_search(query, user_info=None):
    log(f"Starting run_food_search for query: {query}, Timestamp: {time_module.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Load dữ liệu (Giữ nguyên)
    base_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    try:
        rag_df = pd.read_pickle(os.path.join(base_path, 'rag_df.pkl'))
        rag_df['embeddings'] = rag_df['embeddings'].apply(lambda x: np.array(x, dtype=np.float32))
        foods_df = load_data_csv(os.path.join(base_path, 'foods.csv'))
        input_foods_df = load_data_csv(os.path.join(base_path, 'input_foods.csv'))
        food_nutrients_df = load_data_csv(os.path.join(base_path, 'food_nutrients.csv'))
    except Exception as e:
        log(f"Failed to load data: {str(e)}")
        return {"error": "Failed to load data", "details": str(e)}

    # 2. Khởi tạo FoodDatabase (Giữ nguyên)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            food_db = FoodDatabase(rag_df, food_nutrients_df, input_foods_df, foods_df)
    except Exception as e:
        log(f"Failed to initialize FoodDatabase: {str(e)}")
        return {"error": "Failed to initialize FoodDatabase", "details": str(e)}

    # 3. Phân loại Ý định
    intent = classify_intent(query)
    log(f"Intent classified: {intent}")

    # 4. Thực hiện Truy vấn tìm kiếm (RAG)
    final_results = []
    # Chỉ tìm kiếm nếu truy vấn không phải là quá chung chung/không liên quan.
    if intent not in ["GENERAL_OTHER"]:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                results_df = food_db.search_food(query, 60)
        except Exception as e:
            log(f"Error during search_food: {str(e)}")
            return {"error": "Search failed", "details": str(e)}

        # 4. Chuẩn hóa kết quả
        for _, row in results_df.iterrows():
            fdc_id = row['fdcId']
            details = food_db.get_food_details(fdc_id)
            try:
                ing = food_db.get_ingredients(fdc_id)
            except Exception as e:
                log(f"get_ingredients failed for {fdc_id}: {e}")
                ing = []
            final_results.append({
                'fdcId': int(fdc_id),
                'description': row.get('description') if 'description' in row else str(row.get('fdcId')),
                'nutrients': details.get('nutrients') if details else {},
                'ingredients': ing
            })

    # 5. Preprocess
    prepped = preprocess_results(final_results, max_items=40, max_chars=15000, max_chars_per_item=400)
    
    # 6. Chọn Prompt dựa trên Intent
    
    if intent == "PLANNING":
        prompt = generate_detailed_planning_prompt(prepped, query, user_info)
        max_tokens = 2500 # Tăng giới hạn cho kế hoạch chi tiết
    elif intent == "HEALTH_ADVICE" or intent == "FOOD_INFO":
        prompt = generate_health_advice_prompt(prepped, query, user_info)
        max_tokens = 1500
    elif intent == "GENERAL_OTHER":
        prompt = generate_fallback_prompt(prepped, query, user_info)
        max_tokens = 800
    else: # SEARCH_SUMMARY (và mặc định)
        prompt = build_summary_prompt(prepped, query, max_items_in_prompt=3, user_info=user_info)
        max_tokens = 1000

    log(f"Prompt generated (Intent: {intent}), length: {len(prompt)} chars, Tokens: {max_tokens}")
    
    # 7. Gọi LLM và stream kết quả (Chỉ gọi một lần)
    summary = generate_summary_via_hf(prompt, max_tokens=max_tokens, prepped_items=prepped)
    
    if not summary or len(summary.strip()) == 0:
        log("Summary generation failed, using fallback")
        summary = fallback_summary(prepped, query)

    return {
        "status": "success",
        "query": query,
        "summary": summary,
    }

# -----------------------
# CÁC HÀM PROMPT MỚI VÀ ĐƯỢC CHỈNH SỬA
# -----------------------

def generate_health_advice_prompt(prepped_items, query, user_info):
    items_json = json.dumps(prepped_items, ensure_ascii=False, indent=2)
    
    # Format user info rõ ràng hơn
    user_details = ""
    if user_info:
        user_details = f"""
📋 THÔNG TIN SỨC KHỎE NGƯỜI DÙNG:
- Họ tên: {user_info.get('name', 'N/A')}
- Tuổi: {user_info.get('age', 'N/A')} tuổi
- Giới tính: {user_info.get('gender', 'N/A')}
- Chiều cao: {user_info.get('height', 'N/A')} cm
- Cân nặng hiện tại: {user_info.get('weight', 'N/A')} kg
- Cân nặng mục tiêu: {user_info.get('targetWeight', 'N/A')} kg
- Mục tiêu: {user_info.get('target', 'N/A')} (Giảm cân/Tăng cân/Duy trì)
- Mức độ vận động: {user_info.get('exercise', 'N/A')}
- BMR (Chuyển hóa cơ bản): {user_info.get('bmr', 'N/A')} kcal/ngày
- TDEE (Tiêu hao năng lượng): {user_info.get('tdee', 'N/A')} kcal/ngày
- Dị ứng: {', '.join(user_info.get('allergies', [])) if user_info.get('allergies') else 'Không'}
- Bệnh nền: {', '.join(user_info.get('diseases', [])) if user_info.get('diseases') else 'Không'}
- Ghi chú: {user_info.get('note', 'Không có')}
"""
    
    return f"""Bạn là một Chuyên gia Dinh dưỡng và Sức khỏe cá nhân người Việt Nam. 
Hãy tư vấn ngắn gọn, chính xác và hữu ích cho yêu cầu: "{query}"

{user_details}

DANH SÁCH THỰC PHẨM TÌM KIẾM ĐƯỢC (nếu liên quan):
{items_json}

═══════════════════════════════════════════════════════════════════════

HỆ THỐNG TƯ VẤN:

1️⃣ PHÂN TÍCH SỨC KHỎE CÁ NHÂN:
   - Tóm tắt tình trạng sức khỏe hiện tại DỰA TRÊN DỮ LIỆU (Tuổi, Cân nặng, BMR, TDEE, Mục tiêu).
   - Xác định nhu cầu dinh dưỡng cụ thể cho người dùng này.

2️⃣ CẢNH BÁO AN TOÀN (ĐẠO HAM QUAN TRỌNG):
   - Kiểm tra NGAY LẬP TỨC nếu có BẤT KỲ XUNG ĐỘT nào giữa:
     * Danh sách DỊ ỨNG của người dùng
     * Danh sách BỆNH LÝ của người dùng
     * Các thực phẩm trong danh sách JSON
   - Nếu phát hiện xung đột → CẢNh báo RÕNG RÀNG với màu đỏ tâm lý (ví dụ: "⚠️ CẢNH BÁO: Nếu bạn dị ứng lạc, hạn chế/tránh các sản phẩm có chứa lạc").
   - Nếu người dùng có bệnh Thận → Giải thích cần hạn chế Kali, Sodium, Protein cao.
   - Nếu người dùng có bệnh Tim → Hạn chế muối, chất béo bão hòa.
   - Nếu người dùng có Tiểu đường → Hạn chế đường, chọn carb phức tạp.

3️⃣ LỜI KHUYÊN DINH DƯỠNG TỪ THỰC PHẨM CÓ SẴN:
   - Dựa vào DANH SÁCH THỰC PHẨM JSON: gợi ý 2-3 cách sử dụng hoặc món ăn an toàn.
   - Gợi ý cách chế biến phù hợp với bệnh lý (nếu có).
   - Ưu tiên thực phẩm giàu vitamin, khoáng chất phù hợp với tình trạng sức khỏe.

4️⃣ TÍNH TOÁN CALO VÀ DINH DƯỠNG (nếu có TDEE):
   - Nếu mục tiêu là GIẢM CÂN: Gợi ý lượng calo cần thiết ≈ TDEE - 300-500 kcal
   - Nếu mục tiêu là TĂNG CÂN: Gợi ý lượng calo cần thiết ≈ TDEE + 300-500 kcal
   - Nếu mục tiêu là DUY TRÌ: Lượng calo ≈ TDEE

═══════════════════════════════════════════════════════════════════════

🎯 ĐỊNH DẠNG CÂU TRẢ LỜI:
- Viết hoàn toàn bằng TIẾNG VIỆT, thân thiện, dễ hiểu.
- Sử dụng emoji để dễ theo dõi.
- Ưu tiên TUYỆT ĐỐI sự an toàn (Dị ứng & Bệnh lý).
- KHÔNG lặp lại nguyên văn JSON trong trả lời.
- Kết thúc bằng câu hỏi khuyến khích hỏi thêm chi tiết.

LƯU Ý QUAN TRỌNG:
- Nếu không có thông tin nào trong JSON → Hãy dùng kiến thức chung để tư vấn.
- TUYỆT ĐỐI không đưa ra lời khuyên y tế ngoài sức khỏe định mức (ví dụ: không chữa bệnh).
- Luôn ưu tiên sự an toàn hơn là lợi suất dinh dưỡng.
"""

def generate_detailed_planning_prompt(prepped_items, query, user_info):
    items_json = json.dumps(prepped_items, ensure_ascii=False, indent=2)
    
    # Format user info rõ ràng hơn
    user_details = ""
    if user_info:
        user_details = f"""
📋 THÔNG TIN SỨC KHỎE NGƯỜI DÙNG:
- Họ tên: {user_info.get('name', 'N/A')}
- Tuổi: {user_info.get('age', 'N/A')} tuổi
- Giới tính: {user_info.get('gender', 'N/A')}
- Chiều cao: {user_info.get('height', 'N/A')} cm
- Cân nặng hiện tại: {user_info.get('weight', 'N/A')} kg
- Cân nặng mục tiêu: {user_info.get('targetWeight', 'N/A')} kg
- Mục tiêu: {user_info.get('target', 'N/A')} (Giảm cân/Tăng cân/Duy trì)
- Mức độ vận động: {user_info.get('exercise', 'N/A')}
- BMR (Chuyển hóa cơ bản): {user_info.get('bmr', 'N/A')} kcal/ngày
- TDEE (Tiêu hao năng lượng): {user_info.get('tdee', 'N/A')} kcal/ngày
- Dị ứng: {', '.join(user_info.get('allergies', [])) if user_info.get('allergies') else 'Không'}
- Bệnh nền: {', '.join(user_info.get('diseases', [])) if user_info.get('diseases') else 'Không'}
- Ghi chú: {user_info.get('note', 'Không có')}
"""

    return f"""🎯 Bạn là Chuyên gia Dinh dưỡng Cá nhân (AI Nutritionist). 
Nhiệm vụ: LẬP KẾ HOẠCH ĂN UỐNG CHI TIẾT dựa trên dữ liệu người dùng cụ thể và danh sách thực phẩm gợi ý.
Truy vấn: "{query}"

═══════════════════════════════════════════════════════════════════════

1️⃣ THÔNG TIN NGƯỜI DÙNG:
{user_details}

2️⃣ DANH SÁCH THỰC PHẨM CÓ SẴN (Tham khảo để xây dựng kế hoạch):
{items_json}

═══════════════════════════════════════════════════════════════════════

3️⃣ QUY TRÌNH PHÂN TÍCH (STEP-BY-STEP):

📊 BƯỚC 1 - TÍNH TOÁN NHU CẦU:
   - Tính BMI: weight / (height^2 / 10000) = ?
   - Phân loại tình trạng (Thiếu cân/Bình thường/Thừa cân/Béo phì)
   - Nhu cầu calo HÀNG NGÀY dựa trên MỤC TIÊU:
     * Nếu GIẢM CÂN: Calo = TDEE - 300-500 kcal
     * Nếu TĂNG CÂN: Calo = TDEE + 300-500 kcal
     * Nếu DUY TRÌ: Calo = TDEE

🚨 BƯỚC 2 - KIỂM TRA AN TOÀN (TUYỆT ĐỐI QUAN TRỌNG):
   - ✗ LOẠI BỎ NGAY LẬP TỨC: Bất kỳ thực phẩm nào TRÙng với DỊ ỨNG.
   - ✗ CẬM GIẢM: Thực phẩm không phù hợp với BỆNH LÝ.
     * Bệnh Thận → Hạn chế Kali, Sodium, Protein cao
     * Bệnh Tim → Hạn chế Sodium, chất béo bão hòa
     * Tiểu đường → Hạn chế đường đơn giản, ưu tiên glycemic index thấp
     * Cao huyết áp → Hạn chế Sodium, ưu tiên K+, Mg+
   - Nếu phát hiện xung đột → DỪNG NGAY và BÁO CÁO cho người dùng.

✅ BƯỚC 3 - XÂY DỰNG THỰC ĐƠN:
   - Lựa chọn thực phẩm từ JSON SAU KHI LOẠI BỎ những cái không an toàn.
   - Xây dựng thực đơn CÂN BẰNG 3-4 bữa/ngày (Sáng, Trưa, Chiều/Xen, Tối).
   - Mỗi bữa cần bao gồm: Protein, Carb, Chất béo, Vitamin & Khoáng chất.
   - Tính toán calo GẦN ĐÚNG cho mỗi bữa sao cho TỔNG ≈ nhu cầu hàng ngày.

═══════════════════════════════════════════════════════════════════════

4️⃣ ĐỊNH DẠNG CÂU TRẢ LỜI (Trả lời TOÀN TIẾNG VIỆT):

📝 PHẦN 1 - PHÂN TÍCH NHANH
Tóm tắt:
- Tình trạng sức khỏe hiện tại (BMI, mục tiêu)
- Nhu cầu calo hàng ngày (con số cụ thể)
- Khuyến cáo chung (Nếu có bệnh lý, hạn chế gì)
- ⚠️ CẢNH BÁO nếu có xung đột (dị ứng/bệnh)

📝 PHẦN 2 - KẾ HOẠCH CHI TIẾT (3-7 ngày tùy yêu cầu)
Định dạng mỗi bữa:
🍽️ [BUỔI] - [Calo tính toán]
  • Món 1: [Tên] - [Calo/Dinh dưỡng chính] - [Lý do chọn]
  • Món 2: ...
  • Đồ uống: ...

Ví dụ:
🌅 SÁNG (400-450 kcal)
  • Cơm tấm + Trứng chiên: 300 kcal - Giàu protein & carb
  • Rau muống xào: 80 kcal - Giàu sắt, vitamin
  • Nước cam: 50 kcal - Vitamin C

📝 PHẦN 3 - GHI CHÚ GÉP
- Mẹo ăn uống lành mạnh (Cách chế biến, thời gian ăn, v.v.)
- Nước uống khuyến cáo (Nước lạnh, trà xanh, v.v.)
- Lưu ý về kích thước phần ăn
- Gợi ý snack (nếu cần thiết)

═══════════════════════════════════════════════════════════════════════

5️⃣ LƯU Ý QUAN TRỌNG:

✓ Viết hoàn toàn bằng TIẾNG VIỆT.
✓ Sử dụng emoji & Markdown để dễ theo dõi.
✓ TUYỆT ĐỐI kiểm tra Dị ứng & Bệnh lý TRƯỚC khi gợi ý.
✓ Cho con số CALO CỤ THỂ cho mỗi bữa.
✓ KHÔNG lặp lại nguyên văn JSON.
✓ Nếu JSON không đủ thực phẩm → Dùng kiến thức chung để bổ sung nhưng vẫn TUÂN THỦ yêu cầu an toàn.
✓ Kết thúc bằng lời khuyến khích & câu hỏi để tiếp tục tư vấn.

⚠️ TẤT CẢ CHỈ DẨN TRÊN ĐỀU PHẢI TUỲ THEO DỮ LIỆU CÁ NHÂN CỤ THỂ.
"""

def generate_fallback_prompt(prepped_items, query, user_info):
    items_json = json.dumps(prepped_items, ensure_ascii=False, indent=2)
    
    return f"""Bạn là một chuyên gia dinh dưỡng người Việt Nam. Yêu cầu của người dùng: "{query}" không liên quan trực tiếp đến dinh dưỡng hoặc sức khỏe.

THÔNG TIN NGƯỜI DÙNG: {json.dumps(user_info, ensure_ascii=False, indent=2)}

Hãy trả lời bằng Tiếng Việt, thân thiện:

THỪA NHẬN VÀ CHUYỂN HƯỚNG:
Lịch sự giải thích rằng chủ đề không phải lĩnh vực chuyên môn chính của bạn.

ĐỀ XUẤT GIÁN TIẾP:
Nếu có thể liên kết gián tiếp với sức khỏe, gợi ý ngắn gọn (ví dụ: "Nếu bạn hỏi về sức khỏe, tôi có thể tư vấn về chế độ ăn uống").
Hoặc nếu họ muốn tìm thông tin về một thực phẩm cụ thể nào đó, hãy gợi ý hỏi về dinh dưỡng của nó.

KẾT THÚC:
Kết thúc bằng câu hỏi mở để khuyến khích hỏi về dinh dưỡng/sức khỏe cá nhân.

LƯU Ý QUAN TRỌNG:
Trả lời hoàn toàn bằng Tiếng Việt.
Giữ ngắn gọn, tích cực, không vượt quá 200 từ.
"""
# -----------------------
# CHỨC NĂNG CHÍNH
# -----------------------
if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            if sys.argv[1] == '--sum' or sys.argv[1] == '--summarize':
                # Chức năng này dường như không được dùng trong app, chỉ giữ lại
                if len(sys.argv) < 3:
                    raise ValueError("Input file path required")
                input_file = sys.argv[2]
                with open(input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                prompt = data.get('prompt', '').strip()
                if not prompt:
                    raise ValueError("Empty prompt")
                summary = generate_summary_via_hf(
                    prompt,
                    max_tokens=data.get('max_tokens', 2000)
                )
                output = {"status": "success", "summary": summary or "No summary generated"}
            elif sys.argv[1] == '--search':
                if len(sys.argv) < 4 or sys.argv[2] != '--input-file':
                    raise ValueError("Input file path required for --search")
                input_file = sys.argv[3]
                with open(input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                query = data.get('query', '').strip()
                user_info = data.get('user_data', None) # Lấy user_data từ JSON input
                
                if not query:
                    raise ValueError("Empty query")
                
                # CHẠY TÌM KIẾM VÀ TÓM TẮT
                output = run_food_search(query, user_info)
            else:
                output = {"error": "Invalid command"}
        else:
            output = {"error": "No arguments provided"}

        print(json.dumps(output, ensure_ascii=False), flush=True)
    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "type": type(e).__name__,
            "details": f"{traceback.format_exc()}"
        }, ensure_ascii=False), flush=True)