import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from FoodRecognizer import FoodRecognizer
from tqdm import tqdm
import warnings

# Cấu hình tối ưu cho CPU i7-11800H
warnings.filterwarnings("ignore")
torch.set_num_threads(8) 

def evaluate_79_samples(metadata_path="local_test_metadata.csv"):
    # 1. Khởi tạo mô hình và tính Text Embeddings (Cache)
    recognizer = FoodRecognizer()
    recognizer.prepare_dataset() 
    
    # 2. Đọc 79 mẫu đã tải thành công
    df_test = pd.read_csv(metadata_path)
    test_size = len(df_test)
    
    correct_top1, correct_top3 = 0, 0
    latencies = []

    print(f"--- Đang bắt đầu đánh giá trên {test_size} mẫu cục bộ ---")
    
    for idx, row in tqdm(df_test.iterrows(), total=test_size):
        try:
            # Đo thời gian Inference thực tế trên CPU
            start_time = time.time()
            
            # Nhận diện ảnh với Top-K = 3
            results = recognizer.classify_image(
                top_k=3, 
                custom_image=recognizer._load_image(row['local_path'])
            )
            
            latency = time.time() - start_time
            latencies.append(latency)
            
            # So khớp kết quả
            top3_labels = [res['label'] for res in results]
            ground_truth = row['dish_name']
            
            if top3_labels[0] == ground_truth:
                correct_top1 += 1
            if ground_truth in top3_labels:
                correct_top3 += 1
                
        except Exception as e:
            continue
    
    # Tính toán chỉ số %
    valid_count = len(latencies)
    final_top1 = (correct_top1 / valid_count) * 100
    final_top3 = (correct_top3 / valid_count) * 100
    
    # Trực quan hóa kết quả
    plot_results(final_top1, final_top3, latencies, valid_count)

# Chạy hàm đánh giá
evaluate_79_samples()