import os
import time
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from health_tracker import get_user_data, calculate_bmi

# Initialize LLM with Ollama
llm = OllamaLLM(model="deepseek-r1:8b", temperature=0.7, num_ctx=4096, num_predict=5000)

def setup_rag(knowledge_file='./healthyApp/data/raw/health_basics.txt'):
    try:
        if not os.path.exists(knowledge_file):
            raise FileNotFoundError(f"File not found: {knowledge_file}")
        print(f"Loading file: {knowledge_file}")
        loader = TextLoader(knowledge_file)
        documents = loader.load()
        if not documents:
            raise FileNotFoundError("No content loaded from file")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda"}
        )
        vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="./healthyApp/healthyApp/chroma_db")
        return vectorstore
    except Exception as e:
        print(f"Error in setup_rag: {e}")
        raise

rag_store = setup_rag()

def is_medical_query(query, rag_store):
    keywords = ['bmi', 'tdee', 'nhịp tim', 'giấc ngủ', 'sức khỏe', 'dinh dưỡng', 'bệnh', 'tư vấn', 'y tế']
    if any(word in query.lower() for word in keywords):
        return True
    results = rag_store.similarity_search(query, k=1)
    return bool(results and hasattr(results[0], 'metadata') and results[0].metadata.get('score', 0) > 0.5)

def chat_with_health_advisor(query, user_id=1, stream=False):
    try:
        user, latest_data = get_user_data(user_id)
        personal_info = ""
        if latest_data:
            bmi = calculate_bmi(latest_data['weight'], user['height'])
            personal_info = f"Thông tin người dùng: Tuổi {user['age']}, Giới tính {user['gender']}, BMI: {bmi}, Nhịp tim gần nhất: {latest_data['heart_rate']} bpm, Thời gian ngủ: {latest_data['sleep_hours']} giờ."
        
        if is_medical_query(query, rag_store):
            results = rag_store.similarity_search(query, k=1)
            context = "\n".join([doc.page_content for doc in results])
            prompt = f"""
            Bạn là một cố vấn sức khỏe. Dựa trên kiến thức đáng tin cậy: {context}
            Và thông tin cá nhân của người dùng: {personal_info}
            Trả lời câu hỏi: {query}
            Bạn là một trợ lý trả lời chi tiết, cụ thể, không sử dụng thẻ <think> hoặc các bước suy nghĩ, không hiển thị thẻ <think>
            Hãy trả lời một cách tự nhiên, thân thiện, chính xác bằng tiếng Việt. Không được tự tạo thông tin. Nếu không chắc chắn, hãy khuyên người dùng nên gặp bác sĩ và giải thích bằng tiếng Việt.
            """
        else:
            prompt = f"""
            Bạn là một trợ lý sức khỏe. Dựa trên thông tin người dùng: {personal_info}
            Trả lời câu hỏi: {query}
            Bạn là một trợ lý trả lời ngắn gọn, trực tiếp, không sử dụng thẻ <think> hoặc các bước suy nghĩ, không hiển thị thẻ <think>
            Giữ giọng điệu thân thiện và trả lời bằng tiếng Việt.
            """
        
        print(f"Prompt gửi đến Ollama: {prompt}")
        start_time = time.time()
        
        if stream:
            # Stream response and yield each chunk
            response = ""
            for chunk in llm.stream(prompt):
                response += chunk
                print(chunk, end="", flush=True)  # In từng chunk ra console
                yield chunk  # Trả về chunk cho giao diện (như Streamlit)
            print(f"\nThời gian xử lý: {time.time() - start_time} giây")
            return response
        else:
            # Non-streaming response
            response = llm.invoke(prompt)
            print(f"Thời gian xử lý: {time.time() - start_time} giây")
            return response
    except Exception as e:
        print(f"Lỗi trong chat_with_health_advisor: {e}")
        raise

if __name__ == "__main__":
    test_queries = [
        "Làm thế nào để cải thiện giấc ngủ?",
        "Tính diện tích hình tròn bán kính 5cm.",
        "Viết hàm Python kiểm tra số chẵn lẻ.",
        "Hãy cho tôi 1 thực đơn và chế độ tập bulk-up tăng 7 cân trong 3 tháng."
    ]
    for query in test_queries:
        print(f"Query: {query}")
        if query.startswith("Làm thế nào") or query.startswith("Hãy cho tôi"):
            print("Streaming response:")
            for chunk in chat_with_health_advisor(query, stream=True):
                pass  # In đã được xử lý trong hàm
            print("\n")
        else:
            print(f"Response: {chat_with_health_advisor(query)}\n")