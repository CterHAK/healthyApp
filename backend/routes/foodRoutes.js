const express = require("express");
const router = express.Router();
const foodController = require("../controllers/foodController");
const rateLimit = require("express-rate-limit");

// Giới hạn request để chống spam
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 phút
  max: 100,
  message: { error: "Too many requests, please try again later." },
});

// Validate query khi tìm món ăn
const validateSearch = (req, res, next) => {
  const { q } = req.query;
  if (!q || typeof q !== "string" || q.trim().length === 0) {
    return res
      .status(400)
      .json({ error: "Query parameter 'q' must be a non-empty string" });
  }
  next();
};

// ------------------------------
// 🔹 TÌM KIẾM MÓN ĂN BẰNG PYTHON
// ------------------------------
router.get("/search", validateSearch, limiter, foodController.searchFood);

// ------------------------------
// 🔹 NHẬN DIỆN MÓN ĂN TỬ ẢNH
// ------------------------------
router.post("/recognize", foodController.recognizeFood);

// ------------------------------
// 🔹 LƯU THÔNG TIN MÓN ĂN NGƯỜI DÙNG CHỌN
// ------------------------------
router.post("/", foodController.createFood);

// ------------------------------
// 🔹 LẤY DANH SÁCH MÓN ĂN ĐÃ LƯU (CÓ THỂ LỌC THEO EMAIL)
// ------------------------------
router.get("/:email", foodController.getAllFoods);


module.exports = router;