import { API_BASE } from "../utils/constant";

const API_URL = `${API_BASE}/users`;

// 🔹 Tạo user (sau khi account đã được tạo)
export const createUser = async (userData) => {
  const res = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(userData),
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.message || "Không thể tạo user");
  }

  return await res.json();
};

// 🔹 Lấy thông tin user theo email
export const getUserByEmail = async (email) => {
  const res = await fetch(`${API_URL}/${email}`);

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.message || "Không lấy được thông tin user");
  }

  return await res.json();
};
