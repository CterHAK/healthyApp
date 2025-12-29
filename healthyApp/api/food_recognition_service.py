# -*- coding: utf-8 -*-
import sys
import json
import os
import io

# Fix encoding cho Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from healthyApp.foodRecognizer import FoodRecognizer

def recognize_food(image_path):
    """
    Nhan dien thuc pham tu anh
    
    Args:
        image_path (str): Duong dan hoac URL cua anh
    
    Returns:
        dict: Ket qua nhan dien gom:
            - status: "success" hoac "error"
            - predicted_label: Ten mon an du doan
            - info: Thong tin dinh duong, nguyen lieu, khau phan
            - error: Thong bao loi (neu co)
    """
    try:
        sys.stderr.write(f"[INFO] Starting food recognition for: {image_path}\n")
        sys.stderr.flush()
        
        # Khoi tao recognizer
        recognizer = FoodRecognizer(image_path)
        
        # Chuan bi dataset
        sys.stderr.write("[INFO] Preparing dataset...\n")
        sys.stderr.flush()
        recognizer.prepare_dataset()
        
        # Nhan dien hinh anh
        sys.stderr.write("[INFO] Classifying image...\n")
        sys.stderr.flush()
        predicted_label = recognizer.classify_image()
        
        # Trich xuat thong tin
        sys.stderr.write("[INFO] Extracting food information...\n")
        sys.stderr.flush()
        info = recognizer.extract_information(predicted_label)
        
        # Đảm bảo nutrition là object, không phải string
        nutrition_data = info.get("nutrition", {})
        if isinstance(nutrition_data, str):
            try:
                nutrition_data = json.loads(nutrition_data)
            except:
                nutrition_data = {}
        
        return {
            "status": "success",
            "predicted_label": predicted_label,
            "info": {
                "dish_name": predicted_label,
                "ingredients": info.get("ingredients", []),
                "portion_size": info.get("portion_size", ""),
                "nutrition": nutrition_data
            }
        }
    
    except Exception as e:
        sys.stderr.write(f"[ERROR] Error in food recognition: {str(e)}\n")
        sys.stderr.flush()
        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        result = recognize_food(img_path)
        # Output JSON to stdout (not stderr)
        output = json.dumps(result, ensure_ascii=False, indent=2)
        sys.stdout.write(output)
        sys.stdout.write('\n')  # Add newline
        sys.stdout.flush()
    else:
        result = {
            "status": "error",
            "error": "Image path is required"
        }
        output = json.dumps(result, ensure_ascii=False, indent=2)
        sys.stdout.write(output)
        sys.stdout.write('\n')  # Add newline
        sys.stdout.flush()
