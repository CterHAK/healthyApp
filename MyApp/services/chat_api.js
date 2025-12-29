// chat_api.js
import { API_BASE } from "../utils/constant";
const API_URL = `${API_BASE}/chat`;

/**
 * Gửi tin nhắn chatbot đến backend
 * @param {Object} payload - Chứa message, userData, v.v.
 * @returns {Promise<Object>} Response từ API
 */
export const sendChatMessage = async (payload) => {
  try {
    const response = await fetch(`${API_URL}/message`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.message || `HTTP Error: ${response.status}`
      );
    }

    return await response.json();
  } catch (error) {
    console.error("Error in sendChatMessage:", error);
    throw error;
  }
};

/**
 * Lấy lịch sử chat (tùy chọn - nếu backend hỗ trợ)
 * @param {string} email - Email người dùng
 * @returns {Promise<Array>} Danh sách messages
 */
export const getChatHistory = async (email) => {
  try {
    const response = await fetch(`${API_URL}/history/${email}`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      throw new Error("Không thể lấy lịch sử chat");
    }

    return await response.json();
  } catch (error) {
    console.error("Error in getChatHistory:", error);
    throw error;
  }
};

/**
 * Xóa lịch sử chat (tùy chọn)
 * @param {string} email - Email người dùng
 * @returns {Promise<Object>} Kết quả xóa
 */
export const deleteChatHistory = async (email) => {
  try {
    const response = await fetch(`${API_URL}/history/${email}`, {
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      throw new Error("Không thể xóa lịch sử chat");
    }

    return await response.json();
  } catch (error) {
    console.error("Error in deleteChatHistory:", error);
    throw error;
  }
};
