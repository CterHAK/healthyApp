# testToken.py
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
import traceback

load_dotenv()  # Load .env file
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Error: HF_TOKEN not set in .env")
    exit(1)

print(f"Using HF_TOKEN: {HF_TOKEN[:4]}...{HF_TOKEN[-4:]}")

try:
    client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token=HF_TOKEN, timeout=30)
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "Bạn là một chuyên gia dinh dưỡng người Việt Nam. Hãy trả lời bằng tiếng Việt, thân thiện và chi tiết."},
            {"role": "user", "content": "Xin chào, bạn khỏe không?"}
        ],
        max_tokens=50
    )
    print("Response:", response.choices[0].message.content)
except Exception as e:
    print("Error:", str(e))
    print("Full traceback:")
    traceback.print_exc()