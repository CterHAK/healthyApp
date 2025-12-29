const Food = require("../models/FoodRecognizerModel");

// 🥗 Tạo món ăn mới
const saveFood = async (data) => {
  if (!data.email) throw new Error("Email is required");
  if (!data.dish_name) throw new Error("Tên món ăn không được để trống");
  if (!data.image_url) throw new Error("Image URL is required");
  if (!data.day) throw new Error("Ngày không được để trống");
  if (!data.session) throw new Error("Bữa ăn (session) không được để trống");

  // Chuẩn hóa dữ liệu
  const foodData = {
    email: data.email,
    dish_name: String(data.dish_name),
    ingredients: Array.isArray(data.ingredients) ? data.ingredients : [],
    portion_size: String(data.portion_size || ""),
    nutrition: {
      calories_kcal: Number(data.nutrition?.calories_kcal) || 0,
      protein_g: Number(data.nutrition?.protein_g) || 0,
      carbohydrate_g: Number(data.nutrition?.carbohydrate_g) || 0,
      fat_g: Number(data.nutrition?.fat_g) || 0,
    },
    image_url: String(data.image_url),
    day: String(data.day),
    seassion: String(data.session), // ⚡ trùng schema
    is_recognized: Boolean(data.is_recognized || false),
  };

  const record = new Food(foodData);
  const saved = await record.save();
  return saved;
};


// 🍱 Lấy toàn bộ món ăn (hoặc theo email)
const getAllFoods = async (email = null) => {
  const query = email ? { email } : {};
  const records = await Food.find(query).sort({ createdAt: -1 });

  return records.map(r => ({
    ...r.toObject(),
    ingredients: Array.isArray(r.ingredients) ? r.ingredients : [],
    portion_size: String(r.portion_size || ""),
    nutrition: {
      calories_kcal: Number(r.nutrition?.calories_kcal) || 0,
      protein_g: Number(r.nutrition?.protein_g) || 0,
      carbohydrate_g: Number(r.nutrition?.carbohydrate_g) || 0,
      fat_g: Number(r.nutrition?.fat_g) || 0,
    },
  }));
};


// 🍽️ Lấy món ăn theo ngày (và có thể lọc theo email)
const getFoodByDate = async (day, email = null) => {
  const query = email ? { day, email } : { day };
  const records = await Food.find(query).sort({ createdAt: -1 });

  return records.map(r => ({
    ...r.toObject(),
    ingredients: Array.isArray(r.ingredients) ? r.ingredients : [],
    portion_size: String(r.portion_size || ""), // portion_size là string
    nutrition: {
      fat_g: Number(r.nutrition?.fat_g) || 0,
      protein_g: Number(r.nutrition?.protein_g) || 0,
      calories_kcal: Number(r.nutrition?.calories_kcal) || 0,
      carbohydrate_g: Number(r.nutrition?.carbohydrate_g) || 0,
    },
  }));
};

module.exports = {
  saveFood,
  getAllFoods,
  getFoodByDate,
};
