import torch
from PIL import Image
import pandas as pd
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
import requests
from io import BytesIO
import sys
import torch.nn.functional as F

class FoodRecognizer:
    def __init__(self, image_path=None, device=None, batch_size=128):
        # --- 1️⃣ Cấu hình thiết bị ---
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        sys.stderr.write(f"[INFO] Using device: {self.device}\n")
        
        # --- 2️⃣ Load model & processor ---
        sys.stderr.write("[INFO] Loading CLIP model...\n")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # --- 3️⃣ Load ảnh (nếu có) ---
        self.image_path = image_path
        self.raw_image = self._load_image(image_path) if image_path else None

        # --- 4️⃣ Thông số & Cache ---
        self.batch_size = batch_size
        self.df = None
        self.candidate_labels = None
        self.label_embeddings = None  # Cực kỳ quan trọng để chạy nhanh trên CPU

    def _load_image(self, image_path):
        """Load image from local path or URL"""
        if not image_path: return None
        try:
            if isinstance(image_path, str) and image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path, timeout=30)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
            else:
                return Image.open(image_path).convert("RGB")
        except Exception as e:
            sys.stderr.write(f"[ERROR] {str(e)}\n")
            return None

    def prepare_dataset(self):
        """Tải dataset và TÍNH TOÁN TRƯỚC Text Embeddings (Optimization)"""
        print("[INFO] Loading dataset Codatta/MM-Food-100K...", file=sys.stderr, flush=True)
        import warnings
        warnings.filterwarnings("ignore")
        
        ds = load_dataset("Codatta/MM-Food-100K", split="train", streaming=False)
        self.df = ds.to_pandas().dropna(subset=["dish_name", "image_url"]).reset_index(drop=True)
        self.candidate_labels = self.df['dish_name'].unique().tolist()
        
        # --- Tối ưu hóa: Tính Text Embeddings 1 lần duy nhất ---
        sys.stderr.write(f"[INFO] Pre-computing embeddings for {len(self.candidate_labels)} labels...\n")
        text_batch_size = 32
        embeddings_list = []
        
        for i in range(0, len(self.candidate_labels), text_batch_size):
            batch_labels = self.candidate_labels[i : i + text_batch_size]
            inputs = self.clip_processor(text=batch_labels, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                embeds = self.clip_model.get_text_features(**inputs)
                embeds = F.normalize(embeds, p=2, dim=-1)
                embeddings_list.append(embeds.cpu())
        
        self.label_embeddings = torch.cat(embeddings_list, dim=0)
        sys.stderr.write("[INFO] Dataset and Label Embeddings ready.\n")

    def classify_image(self, top_k=1, custom_image=None):
        """Nhận diện ảnh (Sử dụng Cache Embeddings để đạt tốc độ tối đa)"""
        if self.label_embeddings is None:
            raise ValueError("Hãy gọi prepare_dataset() trước để khởi tạo dữ liệu.")

        # Sử dụng ảnh truyền vào hoặc ảnh lúc khởi tạo
        img = custom_image if custom_image else self.raw_image
        if img is None: raise ValueError("Không có ảnh để nhận diện.")

        # Step 1: Image Embedding
        inputs = self.clip_processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embed = self.clip_model.get_image_features(**inputs)
            image_embed = F.normalize(image_embed, p=2, dim=-1).cpu()

        # Step 2: Similarity (Sử dụng ma trận đã tính sẵn)
        similarities = torch.matmul(image_embed, self.label_embeddings.t())
        
        # Lấy Top-K kết quả
        values, indices = similarities.topk(top_k, dim=1)
        
        results = []
        for i in range(top_k):
            idx = indices[0][i].item()
            results.append({
                "label": self.candidate_labels[idx],
                "score": values[0][i].item()
            })

        # Giữ tính tương thích với code cũ (nếu top_k=1 thì trả về chuỗi trực tiếp)
        return results[0]['label'] if top_k == 1 else results

    def extract_information(self, predicted_label):
        """Trích xuất dinh dưỡng từ món ăn"""
        filtered_df = self.df[self.df['dish_name'] == predicted_label]
        if not filtered_df.empty:
            return {
                "ingredients": filtered_df['ingredients'].iloc[0],
                "portion_size": filtered_df['portion_size'].iloc[0],
                "nutrition": filtered_df['nutritional_profile'].iloc[0]
            }
        return {"ingredients": [], "portion_size": "", "nutrition": {}}

if __name__ == "__main__":
    # Test nhanh tính tương thích
    image_path = "banhmi.jpg" # Thay bằng file thật của bạn
    recognizer = FoodRecognizer(image_path)
    recognizer.prepare_dataset()
    
    # Test Top-1 (tương thích cũ)
    label = recognizer.classify_image(top_k=1)
    print(f"Top-1: {label}")
    
    # Test Top-3 (phục vụ phản biện)
    top3 = recognizer.classify_image(top_k=3)
    print(f"Top-3: {top3}")