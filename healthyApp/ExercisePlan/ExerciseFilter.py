import pandas as pd
import ast

class ExerciseFilter:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

        # --- Chuẩn hóa cluster_label (level) ---
        if 'cluster_label' in self.df.columns:
            self.df['cluster_label'] = pd.to_numeric(
                self.df['cluster_label'].astype(str).str.strip(), errors='coerce'
            )

        # --- Chuẩn hóa các cột list ---
        for col in ['combined_muscles', 'equipment']:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(self._to_list_safe)
                self.df[col] = self.df[col].apply(
                    lambda lst: [str(i).strip().lower() for i in lst]
                )
                self.df[col] = self.df[col].apply(
                    lambda lst: ['none'] if not lst else lst
                )

        # Tạo cột lowercase cho tra cứu tên
        if "name" in self.df.columns:
            self.df["name_lower"] = (
                self.df["name"].astype(str).str.lower().str.strip()
            )

    def _to_list_safe(self, x):
        if isinstance(x, str):
            try:
                val = ast.literal_eval(x)
                return val if isinstance(val, list) else [val]
            except:
                return [x.strip()]
        elif isinstance(x, list):
            return x
        elif pd.isna(x):
            return []
        else:
            return [str(x)]

    # ---------------------------------------------------------
    # TRA CỨU BÀI TẬP THEO TÊN
    # ---------------------------------------------------------
    def get_exercise_by_name(self, exercise_name: str):
        ex = exercise_name.lower().strip()
        row = self.df[self.df["name_lower"] == ex]

        if row.empty:
            return None

        row = row.iloc[0]

        # Chuẩn hóa MET
        try:
            MET = float(row.get("MET", 0))
        except:
            MET = 0

        return {
            "name": row.get("name"),
            "MET": MET,
            "video": row.get("video"),
            "instructions": row.get("instructions"),
            "equipment": row.get("equipment"),
            "muscles": row.get("combined_muscles")
        }

    # ---------------------------------------------------------
    # TÍNH CALORIES THEO MET - SETS × REPS
    # ---------------------------------------------------------
    def estimate_calories_by_name_only(self, exercise_name, sets, reps, weight_kg=70, avg_seconds_per_rep=3):

        info = self.get_exercise_by_name(exercise_name)

        if info is None:
            return {
                "status": "error",
                "message": f"Không tìm thấy bài tập: {exercise_name}"
            }

        MET = info["MET"]

        total_reps = sets * reps
        total_seconds = total_reps * avg_seconds_per_rep
        total_hours = total_seconds / 3600

        calories = MET * weight_kg * total_hours

        return {
            "exercise_name": info["name"],
            "sets": sets,
            "reps": reps,
            "MET": MET,
            "total_reps": total_reps,
            "time_seconds": total_seconds,
            "calories": round(calories, 2)
        }

    # Giữ code filter cũ không thay đổi
    def filter_exercises(self, muscle=None, equipment=None, level=None):
        df = self.df.copy()

        if muscle and muscle.lower() not in ["all", ""]:
            muscle = muscle.lower()
            df = df[df['combined_muscles'].apply(lambda lst: muscle in lst)]

        if equipment and equipment.lower() not in ["all", ""]:
            equipment = equipment.lower()
            df = df[df['equipment'].apply(lambda lst: equipment in lst)]

        if level not in [None, "", "all"]:
            try:
                lvl_val = float(level)
                df = df[df["cluster_label"] == lvl_val]
            except:
                pass

        if df.empty:
            return {"count": 0, "results": []}

        cols_to_keep = [
            "name", "category", "combined_muscles", "equipment",
            "cluster_label", "reps", "sets", "MET", "video", "instructions"
        ]
        df = df[[c for c in cols_to_keep if c in df.columns]]

        return {"count": len(df), "results": df.to_dict(orient="records")}
