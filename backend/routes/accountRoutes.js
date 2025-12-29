const express = require("express");
const router = express.Router();
const accountController = require("../controllers/accountController");

// 📘 Lấy danh sách tất cả tài khoản
router.get("/", accountController.getAccounts);

// 📘 Đăng ký tài khoản mới
router.post("/register", accountController.addAccount);

// 📘 Đăng nhập tài khoản
router.post("/login", accountController.loginAccount);

// 📘 Lấy thông tin tài khoản theo email
router.get("/:email", accountController.getAccountByEmail);

// 📘 Cập nhật mật khẩu theo email
router.put("/:email", accountController.updatePassword);

module.exports = router;
