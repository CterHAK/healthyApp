from transformers import pipeline
import warnings
import os

# Tắt warning (tùy chọn)
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Model thay thế: Hỗ trợ tiếng Việt/multilingual, public
model_id = "Qwen/Qwen2.5-1.5B-Instruct"  # Nhẹ, hiểu tiếng Việt tốt

# Tạo pipeline
generator = pipeline("text-generation", model=model_id)

# Prompt: Format cho Instruct model (tăng chất lượng)
prompt = """<|im_start|>user
Viết một câu chuyện cười ngắn về một lập trình viên Python.<|im_end|>
<|im_start|>assistant
"""

# Generate
response = generator(
    prompt,
    max_new_tokens=150,  # Tăng để câu chuyện dài hơn
    temperature=0.9,     # Sáng tạo cao
    do_sample=True,
    pad_token_id=generator.tokenizer.eos_token_id,
    repetition_penalty=1.1  # Tránh lặp từ
)

print("\n=== CÂU CHUYỆN CƯỜI ===")
print(response[0]['generated_text'])