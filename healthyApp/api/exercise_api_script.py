import sys
import os
import json
import pandas as pd
import math
import io

# ⚡ Ép stdout UTF-8 trên Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --------------------- Project root ---------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from healthyApp.ExercisePlan.ExerciseFilter import ExerciseFilter

CSV_PATH = os.path.join(project_root, "healthyApp/ExercisePlan/processed_data_filtered.csv")

# --------------------- Load dữ liệu ---------------------
def load_data():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Không tìm thấy file CSV tại {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    return ExerciseFilter(df)

# --------------------- Fix NaN/Inf để JSON hợp lệ ---------------------
def fix_json_value(obj):
    if isinstance(obj, dict):
        return {k: fix_json_value(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_json_value(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj

# --------------------- Main ---------------------
if __name__ == "__main__":
    # Lấy input
    if not sys.stdin.isatty():
        try:
            input_data = json.load(sys.stdin)
        except json.JSONDecodeError:
            input_data = {}
    else:
        input_data = {
            "action": "calories_name_only",
            "exercise_name": "Push Up",
            "sets": 4,
            "reps": 12,
            "weight_kg": 68
        }

    action = input_data.get("action")
    engine = load_data()
    result = {}

    try:
        # ---------------------------------------------------------
        # 🎯 Tính calories theo sets × reps
        # ---------------------------------------------------------
        if action == "calories_name_only":
            result = engine.estimate_calories_by_name_only(
                input_data.get("exercise_name"),
                int(input_data.get("sets", 0)),
                int(input_data.get("reps", 0)),
                float(input_data.get("weight_kg", 70))
            )

        # ---------------------------------------------------------
        # 🎯 Lọc bài tập theo muscle/equipment/level
        # ---------------------------------------------------------
        elif action == "muscles":
            all_muscles = sorted({m for lst in engine.df['combined_muscles'] for m in lst})
            result = {"status": "success", "muscle_groups": all_muscles}

        elif action == "equipment":
            all_equipment = sorted({e for lst in engine.df['equipment'] for e in lst})
            result = {"status": "success", "equipment_list": all_equipment}

        elif action == "difficulty":
            all_levels = sorted({int(l) for l in engine.df['cluster_label'].dropna() if str(l).isdigit()})
            result = {"status": "success", "difficulty_list": all_levels}
        elif action == "filter":
            result = {"status": "success", **engine.filter_exercises(
                muscle=input_data.get("muscle"),
                equipment=input_data.get("equipment"),
                level=input_data.get("difficulty")
            )}

        # ---------------------------------------------------------
        # 🎯 Tra cứu thông tin bài tập theo tên
        # ---------------------------------------------------------
        elif action == "get_exercise_info":
            ex_info = engine.get_exercise_by_name(input_data.get("exercise_name"))
            if ex_info is None:
                result = {
                    "status": "error",
                    "message": f"Không tìm thấy bài tập: {input_data.get('exercise_name')}"
                }
            else:
                result = {
                    "status": "success",
                    "exercise": ex_info
                }

        else:
            result = {"status": "error", "message": "Sai action"}

    except Exception as e:
        result = {"status": "error", "message": str(e)}

    result = fix_json_value(result)
    print(json.dumps(result, ensure_ascii=False))
