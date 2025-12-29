import json
import pandas as pd
from collections.abc import Mapping
import os
import sys
sys.path.append('../')
#Read data
def read_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)  # f là file object
    return data

# Flattern data
def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, Mapping):
            items.extend(flatten_dict(value, new_key, sep).items())
        else:
            items.append((new_key, value))
    return dict(items)

def solve_data():
    foods_data = []
    nutrients_data = []
    input_foods_data = []
    conversion_factors_data = []
    data = read_data(r'E:\Pj\healthyApp\data\raw\food_data.json')

    for food in data.get('FoundationFoods',[]):
        fdc_id = food.get('fdcId','Unknown')
        food_description = food.get('description','Unknown')

        #Flatern Data
        food_dict = flatten_dict({
            "fdcId": fdc_id,
            "dataType": food.get("dataType"),
            "description": food_description,
            "foodCategory": food.get("foodCategory"),
            "foodClass": food.get("foodClass"),
            "isHistoricalReference": food.get("isHistoricalReference"),
            "ndbNumber": food.get("ndbNumber"),
            "publicationDate": food.get("publicationDate"),
            "foodAttributes": food.get("foodAttributes"),
            "foodPortions": food.get("foodPortions")
        })
        foods_data.append(food_dict)

        # Xử lý foodNutrients
        for nutrient in food.get("foodNutrients", []):
            nutrient_dict = flatten_dict({
                "fdcId": fdc_id,
                "foodDescription": food_description,
                "nutrient": nutrient.get("nutrient"),
                "amount": nutrient.get("amount"),
                "dataPoints": nutrient.get("dataPoints"),
                "min": nutrient.get("min"),
                "median": nutrient.get("median"),
                "max": nutrient.get("max"),
                "foodNutrientDerivation": nutrient.get("foodNutrientDerivation")
            })
            nutrients_data.append(nutrient_dict)

        # Xử lý inputFoods
        for idx, input_food in enumerate(food.get("inputFoods", [])):
            input_food_dict = flatten_dict({
                "fdcId": fdc_id,
                "foodDescription": food_description,
                "inputFoodIndex": idx,
                "inputFood": input_food
            })
            input_foods_data.append(input_food_dict)

        # Xử lý nutrientConversionFactors
        for idx, factor in enumerate(food.get("nutrientConversionFactors", [])):
            factor_dict = flatten_dict({
                "fdcId": fdc_id,
                "foodDescription": food_description,
                "factorIndex": idx,
                "nutrientConversionFactor": factor
            })
            conversion_factors_data.append(factor_dict)

    # Tạo DataFrame
    foods_df = pd.DataFrame(foods_data)
    nutrients_df = pd.DataFrame(nutrients_data)
    input_foods_df = pd.DataFrame(input_foods_data)
    conversion_factors_df = pd.DataFrame(conversion_factors_data)

    # Lưu vào các file CSV
    os.makedirs("../data/processed", exist_ok=True)
    foods_df.to_csv("../data/processed/foods.csv", index=False, encoding="utf-8")
    nutrients_df.to_csv("../data/processed/food_nutrients.csv", index=False, encoding="utf-8")
    input_foods_df.to_csv("../data/processed/input_foods.csv", index=False, encoding="utf-8")
    conversion_factors_df.to_csv("../data/processed/nutrient_conversion_factors.csv", index=False, encoding="utf-8")

    print("Dữ liệu đã được lưu vào các file:")
    print("- foods.csv")
    print("- food_nutrients.csv")
    print("- input_foods.csv")
    print("- nutrient_conversion_factors.csv")






