// chatController.js
const chatService = require("../services/chatService");

/**
 * POST /api/chat/message
 * Xử lý tin nhắn từ người dùng và gọi food_api_script.py
 */
exports.sendMessage = async (req, res) => {
  try {
    const { message, userData } = req.body;

    console.log("[ChatController] Received message:", {
      message,
      userData: userData ? { email: userData.email, name: userData.name } : "none",
    });

    if (!message || !message.trim()) {
      console.warn("[ChatController] Empty message received");
      return res.status(400).json({ error: "Tin nhắn không được để trống" });
    }

    // Gọi service để xử lý tin nhắn
    console.log("[ChatController] Calling chatService.processUserMessage...");
    const result = await chatService.processUserMessage(message, userData);

    console.log("[ChatController] Got result:", {
      success: result.success,
      hasReply: !!result.reply,
      suggestionCount: result.suggestions?.length || 0,
    });

    res.status(200).json({
      success: true,
      reply: result.reply,
      suggestions: result.suggestions || [],
    });
  } catch (error) {
    console.error("[ChatController] Error in sendMessage:", error);
    res.status(500).json({
      error: "Lỗi xử lý tin nhắn",
      details: error.message,
    });
  }
};

/**
 * GET /api/chat/history/:email
 * Lấy lịch sử chat của người dùng (tùy chọn - nếu lưu trữ trong DB)
 */
exports.getChatHistory = async (req, res) => {
  try {
    const { email } = req.params;

    if (!email) {
      return res.status(400).json({ error: "Email không được để trống" });
    }

    // TODO: Implement nếu muốn lưu history trên server
    res.status(200).json({
      success: true,
      message: "Chat history currently stored locally on device",
    });
  } catch (error) {
    console.error("Error in getChatHistory:", error);
    res.status(500).json({
      error: "Lỗi lấy lịch sử",
      details: error.message,
    });
  }
};

/**
 * DELETE /api/chat/history/:email
 * Xóa lịch sử chat (tùy chọn)
 */
exports.deleteChatHistory = async (req, res) => {
  try {
    const { email } = req.params;

    if (!email) {
      return res.status(400).json({ error: "Email không được để trống" });
    }

    // TODO: Implement nếu muốn lưu history trên server
    res.status(200).json({
      success: true,
      message: "Chat history is managed locally on device",
    });
  } catch (error) {
    console.error("Error in deleteChatHistory:", error);
    res.status(500).json({
      error: "Lỗi xóa lịch sử",
      details: error.message,
    });
  }
};
