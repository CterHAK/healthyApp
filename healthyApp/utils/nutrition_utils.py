# utils/nutrition.py

def calculate_bmr(weight, height, age, gender):

    if gender.lower() == "male":
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161


def calculate_tdee(bmr, activity_level):

    factors = {
        "sedentary": 1.2,       # ít vận động
        "light": 1.375,         # vận động nhẹ
        "moderate": 1.55,       # trung bình
        "active": 1.725,        # hoạt động nhiều
        "very_active": 1.9      # vận động cường độ cao
    }
    return bmr * factors.get(activity_level, 1.2)


def calorie_goal(tdee, current_weight, target_weight, weeks=12):
    """
    Tính calories mục tiêu dựa vào mục tiêu cân nặng
    Giả sử: 1kg mỡ ≈ 7700 kcal
    """
    weight_diff = target_weight - current_weight
    total_cal_diff = weight_diff * 7700
    daily_cal_diff = total_cal_diff / (weeks * 7)

    return tdee + daily_cal_diff


def split_macros(calories, ratio=(0.5, 0.25, 0.25)):
    """
    Tính macro theo tỷ lệ: carbs, protein, fat
    carbs: 4 kcal/g
    protein: 4 kcal/g
    fat: 9 kcal/g
    """
    carbs_ratio, protein_ratio, fat_ratio = ratio

    carbs = (calories * carbs_ratio) / 4
    protein = (calories * protein_ratio) / 4
    fat = (calories * fat_ratio) / 9

    return round(carbs), round(protein), round(fat)


if __name__ == "__main__":
    age = 22
    gender = "male"
    weight = 70   # kg
    height = 175  # cm
    activity = "moderate"
    target_weight = 65
    weeks = 12
    # Tính toán
    bmr = calculate_bmr(weight, height, age, gender)
    tdee = calculate_tdee(bmr, activity)
    goal_calories = calorie_goal(tdee, weight, target_weight, weeks)
    carbs, protein, fat = split_macros(goal_calories)

    # Kết quả
    print(f"BMR: {bmr:.0f} kcal")
    print(f"TDEE: {tdee:.0f} kcal")
    print(f"Calories mục tiêu: {goal_calories:.0f} kcal/ngày")
    print(f"Macro gợi ý: {carbs}g Carbs, {protein}g Protein, {fat}g Fat")
