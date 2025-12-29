// chatRoutes.js
const express = require("express");
const chatController = require("../controllers/chatController");

const router = express.Router();

/**
 * POST /api/chat/message
 * Gửi tin nhắn và nhận phản hồi từ chatbot
 */
router.post("/message", chatController.sendMessage);

/**
 * GET /api/chat/history/:email
 * Lấy lịch sử trò chuyện (tùy chọn)
 */
router.get("/history/:email", chatController.getChatHistory);

/**
 * DELETE /api/chat/history/:email
 * Xóa lịch sử trò chuyện (tùy chọn)
 */
router.delete("/history/:email", chatController.deleteChatHistory);

module.exports = router;
