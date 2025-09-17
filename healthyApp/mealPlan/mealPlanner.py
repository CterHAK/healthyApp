import os
import re
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient, login
from utils import load_data_csv
from foodDB import FoodDatabase

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment. Please check your .env file.")

print("HF_TOKEN loaded:", HF_TOKEN[:4] + "...")  # chỉ in một phần token

# ------------------------------
# FastAPI initialization
# ------------------------------
app = FastAPI(
    title="MealPlanner API",
    description="API to generate meal plans based on nutritional queries"
)

# ------------------------------
# Input model
# ------------------------------
class MealPlanRequest(BaseModel):
    query: str
    top_k: int = 30
    days: int = 7
    meals_per_day: int = 3

# ------------------------------
# Query parser
# ------------------------------
def parse_query(query: str) -> tuple[int, int]:
    # Extract number of days
    days_match = re.search(r'(\d+)\s*ngày', query, re.IGNORECASE)
    days = int(days_match.group(1)) if days_match else 1

    # Extract number of meals per day
    meals_match = re.search(r'(\d+)\s*bữa', query, re.IGNORECASE)
    meals_per_day = int(meals_match.group(1)) if meals_match else 3

    return days, meals_per_day

# ------------------------------
# Load datasets
# ------------------------------
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data', 'processed')

    rag_df = pd.read_pickle(os.path.join(data_dir, 'rag_df.pkl'))
    rag_df['embeddings'] = rag_df['embeddings'].apply(lambda x: np.array(x, dtype=np.float32))

    foods_df = load_data_csv(os.path.join(data_dir, 'foods.csv'))
    input_foods_df = load_data_csv(os.path.join(data_dir, 'input_foods.csv'))
    food_nutrients_df = load_data_csv(os.path.join(data_dir, 'food_nutrients.csv'))

except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
    raise

# ------------------------------
# Initialize FoodDatabase
# ------------------------------
food_db = FoodDatabase(rag_df, food_nutrients_df, input_foods_df, foods_df)

# ------------------------------
# MealPlanner class
# ------------------------------
class MealPlanner:
    def __init__(self, model_name='mistralai/Mistral-7B-Instruct-v0.2', prompt=None, hugging_face_token=None):
        self.model_name = model_name
        self.prompt = prompt or ''
        self.hugging_token = hugging_face_token
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.hugging_token:
            login(self.hugging_token)  # login token trực tiếp

    def response(self, query, results_food):
        client = InferenceClient(model=self.model_name, token=self.hugging_token)  # token trực tiếp
        full_prompt = self.prompt + "\n" + results_food.to_string(index=False)
        response = client.chat_completion(
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=1000
        )
        return response.choices[0].message["content"]

# ------------------------------
# Initialize MealPlanner
# ------------------------------
planner = MealPlanner(
    hugging_face_token=HF_TOKEN,
    prompt="Bạn là chuyên gia dinh dưỡng. Hãy soạn thực đơn {days} ngày, mỗi ngày {meals_per_day} bữa (sáng, trưa, tối). Sử dụng các món ăn dưới đây."
)

# ------------------------------
# FastAPI endpoints
# ------------------------------
@app.get("/check_token")
async def check_token():
    """Check if HF_TOKEN is loaded correctly"""
    return {"HF_TOKEN_loaded": bool(HF_TOKEN), "value": HF_TOKEN[:4]+"..."}

@app.get("/")
async def root():
    return {"message": "Welcome to MealPlanner API"}

@app.post("/generate_meal_plan")
async def generate_meal_plan(request: MealPlanRequest):
    try:
        days_parsed, meals_parsed = parse_query(request.query)
        days = days_parsed if days_parsed != 1 else request.days
        meals_per_day = meals_parsed if meals_parsed != 3 else request.meals_per_day

        results_food = food_db.search_food(request.query, request.top_k)

        formatted_prompt = planner.prompt.format(days=days, meals_per_day=meals_per_day)
        planner.prompt = formatted_prompt

        meal_plan = planner.response(query=request.query, results_food=results_food)

        return {
            "query": request.query,
            "top_k_foods": results_food.to_dict(orient="records"),
            "meal_plan": meal_plan,
            "days": days,
            "meals_per_day": meals_per_day
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating meal plan: {str(e)}")

@app.post("/generate_meal_plan_simple")
async def generate_meal_plan_simple(query: str):
    try:
        days, meals_per_day = parse_query(query)
        top_k = 30
        results_food = food_db.search_food(query, top_k)

        formatted_prompt = planner.prompt.format(days=days, meals_per_day=meals_per_day)
        planner.prompt = formatted_prompt

        meal_plan = planner.response(query=query, results_food=results_food)

        return {
            "query": query,
            "top_k_foods": results_food.to_dict(orient="records"),
            "meal_plan": meal_plan,
            "days": days,
            "meals_per_day": meals_per_day
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating meal plan: {str(e)}")
