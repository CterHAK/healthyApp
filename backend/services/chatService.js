// chatService.js
const { spawn } = require("child_process");
const path = require("path");
const os = require("os");
const fs = require("fs");

const PYTHON_SCRIPT = path.join(
  __dirname,
  "../../healthyApp/api/food_api_script.py"
);

/**
 * Chạy Python script với query tìm kiếm
 * @param {string} query - Query tìm kiếm
 * @param {Object} userData - Dữ liệu người dùng
 * @returns {Promise<string>} Kết quả từ Python
 */
function runPythonScript(query, userData = {}) {
  return new Promise((resolve, reject) => {
    // Tạo file input tạm thời
    const tempFile = path.join(os.tmpdir(), `search_${Date.now()}.json`);
    const inputData = {
      query: query,
    };

    try {
      fs.writeFileSync(tempFile, JSON.stringify(inputData, null, 2), {
        encoding: "utf8",
      });
      console.log(`[Python] Created temp file: ${tempFile}`);
    } catch (err) {
      return reject(new Error(`Cannot write temp file: ${err.message}`));
    }

    let output = "";
    let errorOutput = "";

    // Chạy Python script với --search argument
    console.log(
      `[Python] Spawning: python "${PYTHON_SCRIPT}" --search --input-file "${tempFile}"`
    );
    
    const py = spawn("python", [PYTHON_SCRIPT, "--search", "--input-file", tempFile], {
      timeout: 180000,
      maxBuffer: 10 * 1024 * 1024,
    });

    py.stdout.on("data", (data) => {
      const text = data.toString("utf8");
      output += text;
      console.log("[Python stdout]:", text);
    });

    py.stderr.on("data", (data) => {
      const text = data.toString("utf8");
      errorOutput += text;
      console.log("[Python stderr]:", text);
    });

    py.on("close", (code) => {
      console.log(`[Python] Process closed with code: ${code}`);

      // Xóa file tạm
      try {
        fs.unlinkSync(tempFile);
      } catch (e) {
        console.log(`[Python] Cannot delete temp file: ${e.message}`);
      }

      if (code !== 0) {
        console.error(`[Python] Python exited with code ${code}`);
        console.error("[Python] Error output:", errorOutput);
        return reject(
          new Error(
            `Python error: ${errorOutput || `Exit code ${code}`}`
          )
        );
      }

      try {
        // Parse JSON output từ Python
        const lines = output.trim().split("\n");
        let result = null;

        // Tìm JSON object cuối cùng (complete message)
        for (let i = lines.length - 1; i >= 0; i--) {
          const line = lines[i].trim();
          if (!line) continue;

          try {
            const parsed = JSON.parse(line);
            if (
              parsed &&
              (parsed.type === "complete" ||
                parsed.summary ||
                parsed.status === "success")
            ) {
              result = parsed;
              break;
            }
          } catch (e) {
            // Skip invalid JSON
          }
        }

        if (!result) {
          // Fallback: tìm dòng JSON hợp lệ bất kỳ
          for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;
            try {
              const parsed = JSON.parse(line);
              if (parsed && parsed.summary) {
                result = parsed;
                break;
              }
            } catch (e) {
              // Skip
            }
          }
        }

        if (!result) {
          // Nếu vẫn không tìm thấy JSON, trả về output text
          result = {
            summary: output || "Không có kết quả",
            status: "success",
          };
        }

        console.log("[Python] Final result:", result);
        resolve(result);
      } catch (e) {
        console.error("[Python] Parse error:", e);
        reject(e);
      }
    });

    py.on("error", (err) => {
      console.error("[Python] Process error:", err);
      reject(err);
    });
  });
}

/**
 * Xử lý tin nhắn người dùng
 * Phân tích tin nhắn và gọi food_api_script.py nếu cần
 */
exports.processUserMessage = async (userMessage, userData = {}) => {
  try {
    console.log(`[ChatService] Processing message: "${userMessage}"`);

    // Gọi Python script với query
    const pythonResult = await runPythonScript(userMessage, {
      email: userData.email || "guest",
      name: userData.name || "User",
      health_info: userData.health_info || {},
    });

    console.log("[ChatService] Python result:", pythonResult);

    // Xử lý kết quả từ Python
    let reply = "";
    let suggestions = [];

    if (pythonResult.error) {
      reply = pythonResult.error;
    } else if (pythonResult.summary) {
      reply = pythonResult.summary;
    } else if (pythonResult.type === "complete" && pythonResult.summary) {
      reply = pythonResult.summary;
    } else if (pythonResult.status === "success") {
      reply =
        pythonResult.summary ||
        "Không tìm thấy thông tin. Vui lòng thử lại với từ khóa khác.";
    } else {
      reply =
        "Không thể xử lý yêu cầu. Vui lòng thử lại với từ khóa cụ thể hơn.";
    }

    // Gợi ý các câu hỏi tiếp theo dựa trên nội dung tin nhắn
    suggestions = generateSuggestions(userMessage);

    return {
      success: true,
      reply,
      suggestions,
      raw_data: pythonResult,
    };
  } catch (error) {
    console.error("[ChatService] Error in processUserMessage:", error);

    return {
      success: false,
      reply:
        "❌ Lỗi kết nối. Vui lòng kiểm tra và thử lại.\n\nGợi ý: Hãy hỏi về thực phẩm cụ thể như 'cá hồi', 'rau xanh', 'gạo lứt', v.v.",
      suggestions: [
        "Cá hồi có những chất gì?",
        "Thực phẩm tốt cho tim mạch",
        "Món ăn từ rau xanh",
      ],
      error: error.message,
    };
  }
};

/**
 * Tạo gợi ý câu hỏi tiếp theo
 */
function generateSuggestions(userMessage) {
  const suggestions = [];
  const msg = userMessage.toLowerCase();

  // Gợi ý dựa trên nội dung tin nhắn
  if (
    msg.includes("dinh dưỡng") ||
    msg.includes("calo") ||
    msg.includes("đạm")
  ) {
    suggestions.push("Gợi ý thực phẩm thay thế");
  }

  if (msg.includes("dị ứng") || msg.includes("bệnh")) {
    suggestions.push("Thực phẩm an toàn khác");
  }

  if (msg.includes("giảm cân") || msg.includes("tăng cân")) {
    suggestions.push("Kế hoạch ăn chi tiết");
  }

  if (suggestions.length === 0) {
    suggestions.push(
      "Thực phẩm tương tự",
      "Cách chế biến",
      "Kế hoạch ăn hôm nay"
    );
  }

  return suggestions.slice(0, 3);
}

/**
 * Xây dựng context dựa trên thông tin người dùng
 */
function buildUserContext(userData = {}) {
  const health_info = userData.health_info || {};

  let context = "Thông tin người dùng:\n";
  context += `- Tên: ${userData.name || "Không xác định"}\n`;
  context += `- Email: ${userData.email || "Không xác định"}\n`;

  if (health_info.age)
    context += `- Tuổi: ${health_info.age}\n`;
  if (health_info.gender) context += `- Giới tính: ${health_info.gender}\n`;
  if (health_info.weight)
    context += `- Cân nặng: ${health_info.weight}kg\n`;
  if (health_info.height) context += `- Chiều cao: ${health_info.height}cm\n`;
  if (health_info.target) context += `- Mục tiêu: ${health_info.target}\n`;

  if (health_info.allergies && health_info.allergies.length > 0) {
    context += `- Dị ứng: ${health_info.allergies.join(", ")}\n`;
  }

  if (health_info.diseases && health_info.diseases.length > 0) {
    context += `- Bệnh: ${health_info.diseases.join(", ")}\n`;
  }

  return context;
}
