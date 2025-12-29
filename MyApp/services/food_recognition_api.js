// ==========================================
// 🍽️ FOOD RECOGNITION API SERVICE
// ==========================================
// API để gọi chức năng nhận diện thực phẩm từ backend
import { API_BASE } from "../utils/constant";

const API_URL = `${API_BASE}/foods`;

/**
 * 📷 Nhận diện thực phẩm từ ảnh
 * @param {string} imagePath - Đường dẫn hoặc URL của ảnh
 * @returns {Promise<Object>} Kết quả nhận diện
 */
export const recognizeFood = async (imagePath) => {
  try {
    console.log(`[API] Sending request to: ${API_URL}/recognize`);
    
    const response = await fetch(`${API_URL}/recognize`, {
      method: "POST",
      headers: { 
        "Content-Type": "application/json",
        "Connection": "keep-alive"
      },
      body: JSON.stringify({ image_path: imagePath }),
      timeout: 900000  // 15 minutes timeout - CLIP classification needs 12-15 minutes
    });

    console.log(`[API] Response status: ${response.status}`);
    console.log(`[API] Response headers:`, response.headers);
    
    // Get response text first to debug
    const responseText = await response.text();
    console.log(`[API] Response text (first 500 chars): ${responseText.substring(0, 500)}`);
    
    // Try to parse as JSON
    let result;
    try {
      result = JSON.parse(responseText);
    } catch (parseErr) {
      console.error(`[API] JSON parse error: ${parseErr.message}`);
      console.error(`[API] Response was: ${responseText.substring(0, 1000)}`);
      throw new Error(`Invalid response format: ${parseErr.message}. Response: ${responseText.substring(0, 200)}`);
    }

    if (!response.ok) {
      console.error(`[API] HTTP Error ${response.status}:`, result);
      throw new Error(result.message || result.error || `HTTP ${response.status}: Không thể nhận diện thực phẩm`);
    }

    if (result.status === "success" && result.data) {
      console.log(`[API] Success - returning data with dish:`, result.data.predicted_label);
      return result.data;
    } else {
      console.error(`[API] Recognition failed:`, result);
      throw new Error(result.message || result.error || "Nhận diện thất bại");
    }
  } catch (error) {
    console.error("[API] Food recognition error:", error);
    throw error;
  }
};

/**
 * 🔄 Lưu thông tin thực phẩm đã nhận diện
 * @param {Object} foodData
 * @returns {Promise<Object>}
 */
export const saveFoodFromRecognition = async (foodData) => {
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...foodData,
        is_recognized: true, 
      }),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.message || "Không thể lưu thực phẩm");
    }

    return await response.json();
  } catch (error) {
    console.error("Save food error:", error);
    throw error;
  }
};
