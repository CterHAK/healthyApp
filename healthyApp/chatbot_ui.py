import streamlit as st
import requests
import json
import mysql.connector
from mysql.connector import Error

API_URL = "http://localhost:8000/chatbot"

# Cấu hình kết nối MySQL
db_config = {
    'user': 'root',
    'password': '123456',
    'host': 'localhost',
    'database': 'health_tracker'
}

# Thiết lập tiêu đề và mô tả
st.title("Health Chatbot UI")
st.markdown("Hỏi bất kỳ câu hỏi nào về sức khỏe, chatbot sẽ trả lời ngay!")

# Khởi tạo lịch sử trò chuyện và thông tin người dùng trong session_state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_info" not in st.session_state:
    st.session_state.user_info = None

# Nhập User ID
id = st.number_input("User ID", min_value=1, value=1, step=1, key="user_id_input")

# Lấy thông tin người dùng từ MySQL khi id thay đổi
if id != st.session_state.get("last_user_id", None):
    st.session_state.last_user_id = id
    with st.spinner("Đang lấy thông tin người dùng..."):
        try:
            # Kết nối tới MySQL
            connection = mysql.connector.connect(**db_config)
            cursor = connection.cursor(dictionary=True)
            
            # Truy vấn thông tin người dùng
            query = "SELECT id, username, age, gender, height FROM users WHERE id = %s"
            cursor.execute(query, (id,))
            user_info = cursor.fetchone()
            
            if user_info:
                st.session_state.user_info = user_info
                st.success("Đã tải thông tin người dùng!")
            else:
                st.session_state.user_info = None
                st.error(f"Không tìm thấy người dùng với User ID {id}")
                
            cursor.close()
            connection.close()
            
        except Error as e:
            st.session_state.user_info = None
            st.error(f"Lỗi kết nối tới MySQL: {str(e)}")

# Hiển thị thông tin người dùng
st.subheader("Thông tin người dùng")
if st.session_state.user_info:
    user_info = st.session_state.user_info
    st.markdown(f"**Tên**: {user_info.get('name', 'Không có thông tin')}")
    st.markdown(f"**Tuổi**: {user_info.get('age', 'Không có thông tin')}")
    st.markdown(f"**Giới tính**: {user_info.get('gender', 'Không có thông tin')}")
    st.markdown(f"**Chiều cao**: {user_info.get('height', 'Không có thông tin')} m")
else:
    st.markdown("Không có thông tin người dùng hoặc User ID không hợp lệ.")

# Khu vực nhập câu hỏi
query = st.text_area("Nhập câu hỏi của bạn:", height=100)

# Bố cục nút
col1, col2 = st.columns([1, 1])
with col1:
    submit_button = st.button("Gửi")
with col2:
    clear_button = st.button("Xóa lịch sử trò chuyện")

# Xử lý nút xóa lịch sử
if clear_button:
    st.session_state.chat_history = []
    st.success("Lịch sử trò chuyện đã được xóa.")

# Xử lý gửi câu hỏi
if submit_button:
    if not query.strip():
        st.warning("Vui lòng nhập câu hỏi.")
    else:
        # Thêm câu hỏi vào lịch sử
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Hiển thị spinner khi gửi yêu cầu
        with st.spinner("Đang xử lý..."):
            try:
                # Gửi yêu cầu tới API với streaming
                response = requests.post(
                    API_URL,
                    json={"query": query, "id": id},
                    stream=True,  # Bật streaming
                    timeout=1000
                )
                
                if response.status_code == 200:
                    # Khởi tạo container cho phản hồi streaming
                    response_container = st.empty()
                    full_response = ""
                    
                    # Xử lý streaming
                    for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                        if chunk:
                            chunk_text = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk
                            try:
                                # Giả sử mỗi chunk là một JSON chứa trường "response"
                                chunk_data = json.loads(chunk_text)
                                chunk_response = chunk_data.get("response", "")
                                full_response += chunk_response
                                # Cập nhật container với phản hồi tích lũy
                                response_container.markdown(full_response)
                            except json.JSONDecodeError:
                                # Nếu chunk không phải JSON, hiển thị trực tiếp
                                full_response += chunk_text
                                response_container.markdown(full_response)
                    
                    # Thêm phản hồi hoàn chỉnh vào lịch sử
                    st.session_state.chat_history.append({"role": "bot", "content": full_response})
                    st.success("Phản hồi hoàn tất!")
                else:
                    st.error(f"Lỗi: {response.status_code} - {response.text}")
                    st.session_state.chat_history.append({"role": "bot", "content": f"Lỗi: {response.status_code} - {response.text}"})
            
            except requests.exceptions.RequestException as e:
                st.error(f"Lỗi kết nối tới API: {str(e)}")
                st.session_state.chat_history.append({"role": "bot", "content": f"Lỗi kết nối: {str(e)}"})

# Hiển thị lịch sử trò chuyện
st.subheader("Lịch sử trò chuyện")
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**Bạn**: {msg['content']}")
    else:
        st.markdown(f"**Chatbot**: {msg['content']}")