from chatbot import chat_with_health_advisor
from health_tracker import get_user_data, calculate_bmi

def test_health_tracker():
    print("=== Test health_tracker ===")
    user, latest = get_user_data(1)
    print("User:", user)
    print("Latest health data:", latest)
    if latest and user:
        print("BMI:", calculate_bmi(latest['weight'], user['height']))

def test_chatbot():
    print("\n=== Test chatbot ===")
    queries = [
        "Tôi muốn biết BMI của mình.",
        "Làm thế nào để cải thiện giấc ngủ?",
        "Viết hàm Python kiểm tra số chẵn lẻ."
    ]
    for q in queries:
        print(f"Q: {q}")
        print("A:", chat_with_health_advisor(q, user_id=1))

if __name__ == "__main__":
    test_health_tracker()
    test_chatbot()
