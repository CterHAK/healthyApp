const mongoose = require("mongoose");

// 🔹 Schema cho Food từ nhận diện ảnh
const FoodSchema = new mongoose.Schema({
  email: { type: String, required: true }, // Email người dùng
  dish_name: { type: String, required: true }, // Tên món ăn
  ingredients: { type: [String], default: [] }, // Danh sách nguyên liệu
  portion_size: { type: String, default: "" }, // Kích thước khẩu phần
  nutrition: {
    calories_kcal: { type: Number, default: 0 },
    protein_g: { type: Number, default: 0 },
    carbohydrate_g: { type: Number, default: 0 },
    fat_g: { type: Number, default: 0 },
  },
  image_url: { type: String, required: true }, // URL ảnh từ Cloudinary
  day: { type: String, required: true }, // Ngày (DD-MM-YYYY hoặc "Thứ 3")
  seassion: { type: String, required: true }, // Bữa ăn: "Sáng", "Trưa", "Tối", "Bữa nhẹ"
  is_recognized: { type: Boolean, default: false }, // Có phải từ nhận diện tự động không
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
}, { collection: "Food" });

// Index để tìm kiếm nhanh
FoodSchema.index({ email: 1, day: 1 });
FoodSchema.index({ email: 1, createdAt: -1 });

module.exports = mongoose.model("Food", FoodSchema);
