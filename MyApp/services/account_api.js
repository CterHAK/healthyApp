// account_api.js
import { API_BASE } from "../utils/constant";

const API_URL = `${API_BASE}/accounts`;
// -------------------- REGISTER --------------------
export const registerAccount = async (email, password) => {
  const res = await fetch(`${API_URL}/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.message || "Đăng ký thất bại");
  }

  return await res.json();
};

export const loginAccount = async (email, password) => {
  const res = await fetch(`${API_URL}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.message || "Đăng nhập thất bại");
  }

  return await res.json();
};

// -------------------- GET ACCOUNT BY EMAIL --------------------
export const getAccountByEmail = async (email) => {
  const res = await fetch(`${API_URL}/${email}`);

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.message || "Không lấy được thông tin");
  }

  return await res.json();
};
