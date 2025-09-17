from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from mealPlan.mealPlanner import app as mealplan_app
from chatbot import chat_with_health_advisor
from health_tracker import get_user_data, calculate_bmi
import asyncio
import json
import logging
from types import GeneratorType
from collections.abc import AsyncGenerator

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount mealPlan API
app.mount("/mealplan", mealplan_app)

class ChatbotRequest(BaseModel):
    query: str
    user_id: int = 1

async def stream_health_advisor(query: str, user_id: int):
    """
    Generator để stream phản hồi từ chatbot.
    """
    try:
        # Gọi chat_with_health_advisor với stream=True
        response = chat_with_health_advisor(query, user_id=user_id, stream=True)
        logger.debug(f"Response type from chat_with_health_advisor: {type(response)}")

        # Xử lý generator từ chat_with_health_advisor
        if isinstance(response, GeneratorType):
            for chunk in response:
                logger.debug(f"Generator chunk: {chunk}")
                yield json.dumps({"response": chunk})
                await asyncio.sleep(0.05)  # Delay nhỏ để streaming mượt
        elif isinstance(response, AsyncGenerator):
            async for chunk in response:
                logger.debug(f"Async generator chunk: {chunk}")
                yield json.dumps({"response": chunk})
                await asyncio.sleep(0.05)
        else:
            logger.error(f"Unexpected response type: {type(response)}")
            yield json.dumps({"response": f"Lỗi: Kiểu dữ liệu không hỗ trợ: {type(response)}"})

    except Exception as e:
        logger.error(f"Error in stream_health_advisor: {str(e)}")
        yield json.dumps({"response": f"Lỗi: {str(e)}"})

@app.post("/chatbot")
async def chatbot_endpoint(body: ChatbotRequest):
    """
    Gửi câu hỏi tới health advisor chatbot với streaming.
    """
    logger.debug(f"Received query: {body.query}, user_id: {body.user_id}")
    return StreamingResponse(
        stream_health_advisor(body.query, body.user_id),
        media_type="application/json"
    )

@app.get("/user/{user_id}")
def user_info(user_id: int):
    """
    Lấy thông tin user và health data mới nhất.
    """
    user, latest = get_user_data(user_id)
    return {"user": user, "latest": latest}

@app.get("/bmi")
def bmi_endpoint(weight: float = Query(...), height: float = Query(...)):
    """
    Tính BMI nhanh.
    """
    return {"bmi": calculate_bmi(weight, height)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)