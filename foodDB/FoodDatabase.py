import pandas as pd
import numpy as np
import sys

sys.path.append('../')
from utils import load_data_csv, get_faiss_index, get_food_index
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from pulp import *

import torch
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='E:\Pj\healthyApp\.env')

# Tắt cảnh báo symlink
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'true'



class FoodDatabase:
    def __init__(self, rag_df, nutrients_df, input_foods, foods_df, rag_index=None, food_index=None,
                 embedding_model=None):
        self.rag_df = rag_df
        self.nutrients_df = nutrients_df
        self.input_foods = input_foods
        self.foods_df = foods_df

        # Khởi tạo mô hình embedding
        self.model = SentenceTransformer(
            'paraphrase-multilingual-mpnet-base-v2') if embedding_model is None else embedding_model


        # Khởi tạo FAISS index
        self.rag_index = get_faiss_index(rag_df) if rag_index is None else rag_index
        self.food_index = get_food_index(self.foods_df, self.model) if food_index is None else food_index

        self._food_details_cache = {}
        self._build_food_details_cache()

    def _build_food_details_cache(self):
        for fdc_id in self.foods_df['fdcId'].unique():
            food_nutrient_data = self.nutrients_df[self.nutrients_df['fdcId'] == fdc_id]
            if food_nutrient_data.empty:
                continue

            food_description = self.foods_df[self.foods_df['fdcId'] == fdc_id]['description'].iloc[0]
            aggregated_nutrients = {}

            for _, row in food_nutrient_data.iterrows():
                nutrient_name = str(row['nutrient.name']).lower()
                amount = row['amount']
                unit = str(row['nutrient.unitName']).lower()
                derivation = str(row['foodNutrientDerivation.description']).lower()

                if nutrient_name not in aggregated_nutrients:
                    aggregated_nutrients[nutrient_name] = {
                        'amount': amount,
                        'unit': unit,
                        'derivation': [derivation]
                    }
                else:
                    if aggregated_nutrients[nutrient_name]['unit'] == unit:
                        aggregated_nutrients[nutrient_name]['amount'] += amount

            self._food_details_cache[fdc_id] = {
                'description': food_description,
                'nutrients': aggregated_nutrients
            }


    def search(self, query, k):
        """Search theo nutrient (rag_df)"""
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        distance, indices = self.rag_index.search(query_embedding, k)
        top_k_indices = indices[0]
        relevant_chunks_df = self.rag_df.iloc[top_k_indices].copy()
        return relevant_chunks_df

    def search_food(self, query, k):
        """Search thẳng top-k món ăn (foods_df)"""
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        distance, indices = self.food_index.search(query_embedding, k)
        top_foods = self.foods_df.iloc[indices[0]].copy()

        # Lọc món ăn giàu protein, ít calo (nới lỏng điều kiện)
        filtered_foods = []
        for _, row in top_foods.iterrows():
            details = self.get_food_details(row['fdcId'])
            if details and details['nutrients'].get('protein', {}).get('amount', 0) >= 5 and \
                    details['nutrients'].get('energy', {}).get('amount', 0) <= 600:
                filtered_foods.append(row)
        filtered_df = pd.DataFrame(filtered_foods)[['fdcId', 'description']]
        print(f"Filtered foods: {len(filtered_df)} items")
        return filtered_df

    def get_food_details(self, fdc_id):
        return self._food_details_cache.get(fdc_id)

    def get_ingredients(self, fdc_id):
        ingredients_list = []
        ingredients_data = self.input_foods[self.input_foods['fdcId'] == fdc_id].copy()

        if not ingredients_data.empty:
            food_description = self._food_details_cache.get(fdc_id, {}).get('description', f'fdcId {fdc_id}')
            ingredients_list.append(f'---- Ingredients for {food_description}')

            for _, row in ingredients_data.iterrows():
                input_food_description = row.get('inputFood.foodDescription', 'Unknown Ingredient')
                input_food_category = row.get('inputFood.foodCategory.description', 'Unknown category')
                ingredients_list.append(f'- {input_food_description} (Category: {input_food_category})')

        return ingredients_list




if __name__ == '__main__':
    # Load dữ liệu
    rag_df = pd.read_pickle('../data/processed/rag_df.pkl')
    rag_df['embeddings'] = rag_df['embeddings'].apply(lambda x: np.array(x, dtype=np.float32))
    foods_df = load_data_csv('../data/processed/foods.csv')
    input_foods_df = load_data_csv('../data/processed/input_foods.csv')
    food_nutrients_df = load_data_csv('../data/processed/food_nutrients.csv')

    # Khởi tạo FoodDatabase
    food_db = FoodDatabase(rag_df, food_nutrients_df, input_foods_df, foods_df)


    query_italy = "Tạo 7 ngày ăn đầy đủ chất dinh dưỡng"
    results_food_italy = food_db.search_food(query_italy, 30)
    print("\n🍽️ Top 5 món ăn Italy:")
    print(results_food_italy.to_string(index=False))

    for _, row in results_food_italy.iterrows():
        fdc_id = row['fdcId']
        print(f"\n👉 Chi tiết món: {row['description']} (fdcId={fdc_id})")
        print("⚡ Nutrients:")
        for nutrient, values in food_db.get_food_details(fdc_id)['nutrients'].items():
            print(f"- {nutrient}: {values['amount']} {values['unit']}")
        print("\n🥗 Ingredients:")
        for ing in food_db.get_ingredients(fdc_id):
            print(ing)