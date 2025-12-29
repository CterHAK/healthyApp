import os
import requests
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def download_test_set(target_count=1000):
    # Lấy mẫu lớn hơn (ví dụ 1000 ) để trừ hao những link hỏng
    sample_size = target_count * 10 
    
    ds = load_dataset("Codatta/MM-Food-100K", split="train")
    df = ds.to_pandas().dropna(subset=["dish_name", "image_url"])
    test_df = df.sample(n=sample_size, random_state=42)

    local_data = []
    # Giả lập trình duyệt hiện đại để tránh bị chặn
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    print(f"[INFO] Đang cố gắng tìm đủ {target_count} ảnh hợp lệ...")
    pbar = tqdm(total=target_count)
    
    for idx, row in test_df.iterrows():
        if len(local_data) >= target_count:
            break # Dừng khi đã đủ số lượng mục tiêu
            
        try:
            # Giảm timeout xuống 5s để bỏ qua nhanh các link chậm
            response = requests.get(row['image_url'], timeout=5, headers=headers)
            if response.status_code == 200:
                img_name = f"food_{idx}.jpg"
                img_path = os.path.join("test_images", img_name)
                
                with open(img_path, 'wb') as f:
                    f.write(response.content)
                
                local_data.append({"dish_name": row['dish_name'], "local_path": img_path})
                pbar.update(1)
        except:
            continue
    pbar.close()
    pd.DataFrame(local_data).to_csv("local_test_metadata.csv", index=False)

if __name__ == "__main__":
    download_test_set(1000) # Thử 1000 mẫu trước