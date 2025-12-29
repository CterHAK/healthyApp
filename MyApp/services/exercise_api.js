
import { API_BASE } from "../utils/constant";

const API_URL = `${API_BASE}/exercises`;
// Lấy danh sách cơ
export const getMuscles = async () => 
  (await fetch(`${API_URL}/muscles`)).json();

// Lấy danh sách thiết bị
export const getEquipment = async () => 
  (await fetch(`${API_URL}/equipment`)).json();

// Lọc bài tập
export const filterExercises = async (params) =>
  (await fetch(`${API_URL}/filter?${new URLSearchParams(params)}`)).json();

// Tra cứu thông tin bài tập
export const getExerciseInfo = async (exercise_name) =>
  (await fetch(`${API_URL}/exercise-info?exercise_name=${encodeURIComponent(exercise_name)}`)).json();

// Tính calories
export const calculateCalories = async (data) =>
  (await fetch(`${API_URL}/calculate-calories`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })).json();

// Thêm nhiều bài tập vào nhóm
export const addExerciseGroupAndDetail = async (data) =>
  (await fetch(`${API_URL}/group`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })).json();

// Lấy nhóm bài tập (có thể filter bằng params: email)
export const getExerciseGroup = async (params = {}) =>
  (await fetch(`${API_URL}/group?${new URLSearchParams(params)}`)).json();

// POST: Thêm plan vào group
export const addExercisePlan = async (data) => {
  // data: { email, group_name, day, session, done_flag }
  return await fetch(`${API_URL}/plan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  }).then(res => res.json());
};

// GET: Lấy plan theo email, có thể filter thêm theo day
export const getExercisePlans = async (params = {}) => {
  // params: { email, day? }
  const query = new URLSearchParams(params).toString();
  return await fetch(`${API_URL}/plan?${query}`)
    .then(res => res.json());
};

// POST: Cập nhật done_flag cho 1 plan
export const markWorkoutDone = async (body = {}) => {
  // body: { email, group_name, day, session }
  return await fetch(`${API_URL}/plan/done`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  }).then(res => res.json());
};

export const getCalorieChartData = async (email, days, weight = 70) => {
  const res = await fetch(
    `${API_URL}/chart-data?email=${email}&days=${days}&weight=${weight}`
  );
  return res.json();
};


export const getCalorieStats = async (email, days, weight = 70) => {
  const res = await fetch(
    `${API_URL}/nutrition-stats?email=${email}&days=${days}&weight=${weight}`
  );
  return res.json();
};

/**
 * Lấy dữ liệu calo & năng lượng tiêu hao của ngày hôm nay
 * @param {string} email - Email người dùng
 * @param {number} weight - Cân nặng hiện tại (kg)
 * @param {string} customDate - Ngày tùy chọn (format YYYY-MM-DD), mặc định là hôm nay
 * @returns {Promise<Object|null>}
 */
export const fetchTodayCalories = async (email, weight = 70, customDate) => {
  if (!email) {
    console.warn("[nutritionApi] Thiếu email");
    return null;
  }

  try {
    // Tạo ngày theo local timezone (giống cách backend xử lý)
    const dateObj = customDate ? new Date(customDate) : new Date();
    const year = dateObj.getFullYear();
    const month = String(dateObj.getMonth() + 1).padStart(2, "0");
    const day = String(dateObj.getDate()).padStart(2, "0");
    const dateStr = `${year}-${month}-${day}`;

    console.log(`[nutritionApi] Fetching daily nutrition → date=${dateStr}, email=${email}`);

    const response = await fetch(
      `${API_URL}/daily-nutrition?email=${encodeURIComponent(email)}&date=${dateStr}&weight=${weight}`
    );

    // Kiểm tra HTTP status
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();

    if (data.status === "success") {
      return data; // { intake, burned, net_calories, ... }
    } else {
      console.warn("[nutritionApi] API không thành công:", data);
      return null;
    }
  } catch (error) {
    console.error("[nutritionApi] Lỗi khi fetch daily nutrition:", error);
    return null;
  }
};