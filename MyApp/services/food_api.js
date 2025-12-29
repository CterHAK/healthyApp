import { API_BASE } from "../utils/constant";

const API_URL = `${API_BASE}/foods`;
// -------------------- LƯU MÓN ĂN --------------------
export const saveFood = async (foodData) => {
  const res = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(foodData),
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.message || "Không thể lưu món ăn");
  }

  return await res.json();
};

// -------------------- LẤY DANH SÁCH MÓN ĂN THEO EMAIL --------------------
export const getAllFoods  = async (email) => {
  const res = await fetch(`${API_URL}/${email}`);

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.message || "Không lấy được danh sách món ăn");
  }

  return await res.json();
};

