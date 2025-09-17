import mysql.connector
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Kết nối MySQL
db_config = {
    'user': 'root',
    'password': '123456',  # Thay bằng password MySQL của bạn
    'host': 'localhost',
    'database': 'health_tracker'
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

def calculate_bmi(weight, height):
    if height <= 0 or weight <= 0:
        return "Dữ liệu không hợp lệ"
    bmi = weight / (height ** 2)
    if bmi < 18.5:
        return f"BMI: {bmi:.2f} (Gầy)"
    elif 18.5 <= bmi < 25:
        return f"BMI: {bmi:.2f} (Bình thường)"
    elif 25 <= bmi < 30:
        return f"BMI: {bmi:.2f} (Thừa cân)"
    else:
        return f"BMI: {bmi:.2f} (Béo phì)"

def get_user_data(user_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    cursor.execute("SELECT * FROM health_data WHERE user_id = %s ORDER BY date DESC LIMIT 1", (user_id,))
    latest_data = cursor.fetchone()
    conn.close()
    return user, latest_data