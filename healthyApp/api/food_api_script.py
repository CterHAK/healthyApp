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

# 1. TÍNH TOÁN ĐƯỜNG DẪN GỐC DỰ ÁN (PROJECT_ROOT)
# __file__ là food_api_script.py
# BASE_DIR là thư mục hiện tại: ...\healthyApp\api
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cần đi ngược 2 cấp để đến thư mục chứa 'healthyApp' và thư mục gốc 'projectHealthy'
PROJECT_ROOT = os.path.join(BASE_DIR, '..', '..')

# 2. THÊM THƯ MỤC GỐC DỰ ÁN VÀO sys.path
sys.path.append(PROJECT_ROOT)

# 3. IMPORT CÁC MODULE CON
from healthyApp.foodDB import FoodDatabase
from healthyApp.utils import load_data_csv, get_faiss_index, get_food_index

# Tải biến môi trường (.env nằm ở D:\TLCN\projectHealthy\projectHealthy\.env)
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, '.env'))

# Tắt cảnh báo symlink
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'true'

# (Bổ sung: client summary via Hugging Face Inference API nếu có token)
try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

HF_TOKEN = os.getenv("HF_TOKEN")  # optional, nếu có sẽ dùng để gọi HF Inference

# -----------------------
# PREPROCESS / DEDUPE / TRUNCATE
# -----------------------
def normalize_name(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\(.*?\)", "", s)              # loại bỏ nội dung trong ()
    s = re.sub(r"[^a-z0-9]+", " ", s)          # chỉ giữ ký tự alnum
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

def build_summary_prompt(prepped_items, query, max_items_in_prompt=8):
    if not prepped_items:
        items = []
    else:
        items = prepped_items[:max_items_in_prompt]

    try:
        items_json = json.dumps(items, ensure_ascii=False, indent=2)
    except Exception:
        items_json = str(items)

    instructions = (
        "Bạn là một trợ lý ẩm thực. Dưới đây là dữ liệu JSON gồm các thực phẩm/nguyên liệu (mỗi phần tử có fdcId, name, description, nutrients, ingredients).\n\n"
        "JSON_DANH_SACH:\n"
        f"{items_json}\n\n"
        "Yêu cầu:\n"
        "1) Viết một câu trả lời bằng TIẾNG VIỆT, thân thiện và dễ hiểu cho người dùng dựa CHÍNH XÁC trên dữ liệu JSON ở trên.\n"
        "2) Bắt đầu bằng một tóm tắt ngắn (1-2 câu) nêu tổng quan những lựa chọn phù hợp với truy vấn: "
        f"\"{query}\".\n"
        "3) Đưa ra 2-4 gợi ý cụ thể cách sử dụng hoặc món ăn có thể làm từ các nguyên liệu/thực phẩm này.\n"
        "4) Nếu cần, nêu vài điểm dinh dưỡng chính (dựa trên trường nutrients khi có) nhưng KHÔNG suy diễn thông tin không có trong JSON.\n"
        "5) Kết thúc bằng một câu hỏi ngắn khuyến khích người dùng: nếu họ muốn kế hoạch ăn uống hoặc công thức chi tiết, hãy hỏi rõ.\n\n"
        "Lưu ý: KHÔNG lặp lại nguyên văn JSON trong phần trả lời, chỉ dựa trên dữ liệu để viết câu trả lời hợp lý.\n"
    )

    return instructions

def generate_summary_via_hf(prompt, model_name=None, max_tokens=2000, prepped_items=None):
    try:
        client = InferenceClient(model=model_name or "mistralai/Mixtral-8x7B-Instruct-v0.1")
        log(f"Starting HF API call with prompt length: {len(prompt)} chars, Timestamp: {time_module.strftime('%Y-%m-%d %H:%M:%S')}")

        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            stream=True
        )

        summary = ''
        chunk_buffer = ''
        min_chunk_size = 100  # Mỗi chunk tối thiểu 100 ký tự

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
# -----------------------
# RUN SEARCH (CHÍNH)
# -----------------------
def run_food_search(query):
    log(f"Starting run_food_search for query: {query}, Timestamp: {time_module.strftime('%Y-%m-%d %H:%M:%S')}")
    # 1. Load dữ liệu
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

    # 2. Khởi tạo FoodDatabase
    try:
        with contextlib.redirect_stdout(sys.stderr):
            food_db = FoodDatabase(rag_df, food_nutrients_df, input_foods_df, foods_df)
    except Exception as e:
        log(f"Failed to initialize FoodDatabase: {str(e)}")
        return {"error": "Failed to initialize FoodDatabase", "details": str(e)}

    # 3. Thực hiện truy vấn
    try:
        with contextlib.redirect_stdout(sys.stderr):
            results_df = food_db.search_food(query, 60)
    except Exception as e:
        log(f"Error during search_food: {str(e)}")
        return {"error": "Search failed", "details": str(e)}

    # 4. Chuẩn hóa kết quả
    final_results = []
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
    
    prompt = build_summary_prompt(prepped, query, max_items_in_prompt=3)
    log(f"Prompt generated, length: {len(prompt)} chars, Timestamp: {time_module.strftime('%Y-%m-%d %H:%M:%S')}")
    summary = generate_summary_via_hf(prompt, max_tokens=1000, prepped_items=prepped)
    if "kế hoạch" in query.lower() or "thực đơn" in query.lower() or "dinh dưỡng chi tiết" in query.lower():
        prompt = generate_detailed_prompt(prepped, query)  # Prompt gốc
    elif "sức khỏe" in query.lower() or "ăn uống" in query.lower() or "tư vấn" in query.lower():
        prompt = generate_health_advice_prompt(prepped, query)  # Prompt sức khỏe
    else:
        prompt = generate_fallback_prompt(prepped, query)  # Prompt fallback
    
    summary = generate_summary_via_hf(prompt, max_tokens=2000, prepped_items=prepped)  # Tăng max_tokens nếu cần
    if not summary or len(summary.strip()) == 0:
        log("Summary generation failed, using fallback")
        summary = fallback_summary(prepped, query)

    return {
        "status": "success",
        "query": query,
        "summary": summary,
    }

def generate_health_advice_prompt(prepped_items, query):
    items_json = json.dumps(prepped_items, ensure_ascii=False, indent=2)
    
    return f"""Bạn là một chuyên gia dinh dưỡng người Việt Nam. Hãy tư vấn ngắn gọn và hữu ích cho yêu cầu: "{query}"




Dựa vào danh sách thực phẩm sau:
{items_json}
Hãy trả lời bằng Tiếng Việt, bao gồm:

PHÂN TÍCH NGẮN GỌN:


Giải thích lợi ích hoặc rủi ro sức khỏe liên quan đến yêu cầu.
Sử dụng thông tin dinh dưỡng từ JSON (như calo, protein, chất béo) để hỗ trợ.


LỜI KHUYÊN THỰC TẾ:


Đưa ra 2-3 mẹo ăn uống lành mạnh dựa trên thực phẩm có sẵn.
Gợi ý cách sử dụng hoặc thay thế để cân bằng dinh dưỡng.


LƯU Ý:


Những điều cần tránh để bảo vệ sức khỏe.
Kết thúc bằng câu hỏi mở để hỏi thêm chi tiết.

LƯU Ý QUAN TRỌNG:

Trả lời hoàn toàn bằng Tiếng Việt.
Giữ ngắn gọn, dễ hiểu, không vượt quá 300 từ.
Tập trung vào tính thực tế và khả thi.
Sử dụng thông tin từ JSON, KHÔNG suy diễn.
Nếu không liên quan đến dữ liệu, hãy lịch sự thừa nhận và gợi ý điều chỉnh yêu cầu.
"""

def generate_detailed_prompt(prepped_items, query):
    items_json = json.dumps(prepped_items, ensure_ascii=False, indent=2)
    
    return f"""Bạn là một chuyên gia dinh dưỡng người Việt Nam. Hãy tư vấn chi tiết cho yêu cầu: "{query}"

Dựa vào danh sách thực phẩm sau:
{items_json}

Hãy tạo một kế hoạch dinh dưỡng CHI TIẾT bằng Tiếng Việt, bao gồm:

1. PHÂN TÍCH YÊU CẦU VÀ MỤC TIÊU:
- Phân tích chi tiết mục tiêu của người dùng
- Ước tính thời gian và các mốc cần đạt được
- Xác định nhu cầu dinh dưỡng cụ thể

2. KẾ HOẠCH DINH DƯỠNG CHI TIẾT:
- Phân bổ bữa ăn trong ngày
- Danh sách thực phẩm được đề xuất và lý do chọn
- Cách kết hợp các món ăn để tối ưu dinh dưỡng

3. HƯỚNG DẪN THỰC HIỆN:
- Cách chuẩn bị và chế biến từng món
- Thời điểm ăn uống phù hợp
- Khẩu phần và số lượng cụ thể

4. LƯU Ý VÀ ĐỀ XUẤT:
- Những điều cần tránh
- Cách điều chỉnh theo nhu cầu cá nhân
- Các hoạt động hỗ trợ (như tập luyện)

Hãy viết với giọng điệu thân thiện, dễ hiểu và khuyến khích. Đưa ra lời khuyên thực tế dựa trên các thực phẩm có sẵn trong danh sách.
Kết thúc bằng câu hỏi mở để tương tác với người dùng.

LƯU Ý QUAN TRỌNG:
- Trả lời hoàn toàn bằng Tiếng Việt
- Tập trung vào tính thực tế và khả thi
- Đưa ra hướng dẫn cụ thể và chi tiết
- Sử dụng thông tin dinh dưỡng từ dữ liệu JSON
- KHÔNG suy diễn thông tin không có trong dữ liệu
"""
def generate_fallback_prompt(prepped_items, query):
    items_json = json.dumps(prepped_items, ensure_ascii=False, indent=2)
    
    return f"""Bạn là một chuyên gia dinh dưỡng người Việt Nam. Yêu cầu của người dùng: "{query}" không liên quan đến dinh dưỡng hoặc sức khỏe.

Hãy trả lời bằng Tiếng Việt, thân thiện:

THỪA NHẬN YÊU CẦU:


Lịch sự giải thích rằng chủ đề không phải lĩnh vực chuyên môn của bạn.


ĐỀ XUẤT HƯỚNG DẪN:


Nếu có thể liên kết với dinh dưỡng, gợi ý ngắn gọn (ví dụ: "Nếu bạn hỏi về sức khỏe, tôi có thể tư vấn về chế độ ăn uống").
Dựa vào danh sách thực phẩm nếu liên quan gián tiếp:
{items_json}
Nếu không, gợi ý đặt lại câu hỏi về ăn uống/sức khỏe.


KẾT THÚC:


Kết thúc bằng câu hỏi mở để khuyến khích hỏi về dinh dưỡng.

LƯU Ý QUAN TRỌNG:

Trả lời hoàn toàn bằng Tiếng Việt.
Giữ ngắn gọn, tích cực, không vượt quá 200 từ.
KHÔNG trả lời nội dung không liên quan, tránh suy diễn.
Nếu không có dữ liệu phù hợp, tập trung vào redirect.
"""
if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            if sys.argv[1] == '--sum' or sys.argv[1] == '--summarize':
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
                if not query:
                    raise ValueError("Empty query")
                output = run_food_search(query)
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