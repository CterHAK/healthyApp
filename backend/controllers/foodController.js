const os = require("os");
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const foodService = require("../services/foodService");

const PYTHON_SCRIPT_PATH = path.join(__dirname, "..", "api", "../../healthyApp/api/food_api_script.py");
const PYTHON_RECOGNITION_SCRIPT = path.join(__dirname, "..", "api", "../../healthyApp/api/food_recognition_service.py");

// ------------------------------
// 🔹 1. TÌM KIẾM MÓN ĂN BẰNG PYTHON
// ------------------------------
const searchFood = async (req, res) => {
  try {
    const { q } = req.query;
    if (!q || typeof q !== "string" || q.trim().length === 0) {
      return res.status(400).json({ error: "Query parameter 'q' must be a non-empty string" });
    }

    console.log(`Processing query: ${q}, Timestamp: ${new Date().toISOString()}`);
    const tempFile = path.join(os.tmpdir(), `search_${Date.now()}.json`);
    fs.writeFileSync(tempFile, JSON.stringify({ query: q.trim() }, null, 2), { encoding: "utf8" });

    // SSE headers
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.flushHeaders();

    const pythonProcess = spawn("python", [PYTHON_SCRIPT_PATH, "--search", "--input-file", tempFile], {
      timeout: 180000,
      encoding: "utf8",
      maxBuffer: 10 * 1024 * 1024,
    });

    let buffer = "";

    pythonProcess.stdout.on("data", (data) => {
      buffer += data;
      let lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const msg = JSON.parse(line);
          if (msg.type === "chunk") {
            res.write(`data: ${JSON.stringify({ chunk: msg.content })}\n\n`);
          } else if (msg.type === "complete") {
            res.write(`data: ${JSON.stringify({ complete: true, summary: msg.summary })}\n\n`);
            res.end();
          } else if (msg.type === "error") {
            res.write(`data: ${JSON.stringify({ error: msg.error })}\n\n`);
            res.end();
          }
        } catch (err) {
          console.error("Invalid JSON from Python:", line);
        }
      }
    });

    pythonProcess.stderr.on("data", (data) => {
      console.error("Python stderr:", data.toString());
      res.write(`data: ${JSON.stringify({ stderr: data.toString() })}\n\n`);
    });

    pythonProcess.on("close", (code) => {
      try {
        fs.unlinkSync(tempFile);
      } catch (e) {}
      console.log(`Python process exited with code: ${code}`);
      res.end();
    });

    req.on("close", () => {
      pythonProcess.kill();
      res.end();
    });
  } catch (error) {
    console.error("Error:", error);
    res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
    res.end();
  }
};

// ------------------------------
// 🔹 2. LƯU KẾT QUẢ MÓN ĂN VÀO MONGODB
// ------------------------------
// 🥗 POST /api/food — Tạo món ăn mới
const createFood = async (req, res) => {
  try {
    const foodData = req.body;

    const savedFood = await foodService.saveFood(foodData);

    return res.status(201).json({
      status: "success",
      message: "Food saved successfully",
      data: savedFood,
    });
  } catch (error) {
    console.error("[ERROR] createFood:", error.message);
    return res.status(400).json({
      status: "error",
      message: error.message,
    });
  }
};


// 🍽️ GET /api/food/by-date?day=2025-10-28&email=abc@gmail.com
const getFoodsByDate = async (req, res) => {
  try {
    const { day, email } = req.query;
    if (!day) {
      return res.status(400).json({ message: "Thiếu tham số 'day'" });
    }

    const foods = await foodService.getFoodByDate(day, email);
    res.status(200).json(foods);
  } catch (error) {
    console.error("❌ Error fetching foods by date:", error);
    res.status(500).json({ message: "Lỗi khi lấy món ăn theo ngày" });
  }
};

// 📷 POST /api/foods/recognize — Nhận diện thực phẩm từ ảnh
const recognizeFood = async (req, res) => {
  try {
    const { image_path } = req.body;
    
    if (!image_path) {
      return res.status(400).json({ 
        status: "error", 
        message: "Image path is required" 
      });
    }

    console.log(`[INFO] Recognizing food from: ${image_path}`);

    // Chạy Python script nhận diện
    const pythonProcess = spawn("python", [PYTHON_RECOGNITION_SCRIPT, image_path], {
      timeout: 1800000, // 30 minutes - CLIP classification on CPU is slow
      encoding: "utf8",
      maxBuffer: 100 * 1024 * 1024, // 100MB for large outputs
    });

    let output = "";
    let errorOutput = "";
    let processTimeout = null;
    let lastActivityTime = Date.now();

    pythonProcess.stdout.on("data", (data) => {
      output += data.toString();
      lastActivityTime = Date.now();
      const snippet = data.toString().substring(0, 100);
      console.log(`[STDOUT] ${snippet}`);
      
      // Reset timeout on data received (keep-alive)
      if (processTimeout) clearTimeout(processTimeout);
      processTimeout = setTimeout(() => {
        console.warn("[WARN] No stdout data received for 20 minutes, killing process");
        pythonProcess.kill();
      }, 20 * 60 * 1000); // 20 minutes without data
    });

    pythonProcess.stderr.on("data", (data) => {
      const dataStr = data.toString();
      errorOutput += dataStr;
      lastActivityTime = Date.now();
      const snippet = dataStr.substring(0, 100);
      console.log(`[STDERR] ${snippet}`);
      
      // Also reset timeout on stderr (process is alive even if stdout is quiet)
      if (processTimeout) clearTimeout(processTimeout);
      processTimeout = setTimeout(() => {
        console.warn("[WARN] No data received for 20 minutes, killing process");
        pythonProcess.kill();
      }, 20 * 60 * 1000); // 20 minutes without data
    });

    pythonProcess.on("error", (err) => {
      console.error(`[ERROR] Process error: ${err.message}`);
      if (processTimeout) clearTimeout(processTimeout);
      if (!res.headersSent) {
        res.status(500).json({
          status: "error",
          message: "Process error during recognition",
          details: err.message
        });
      }
    });

    pythonProcess.on("exit", (code, signal) => {
      console.log(`[INFO] Process exit - code: ${code}, signal: ${signal}`);
      if (signal === "SIGTERM" || signal === "SIGKILL") {
        console.error(`[ERROR] Process was killed with signal: ${signal}`);
        if (!res.headersSent) {
          res.status(500).json({
            status: "error",
            message: `Process killed: ${signal}`,
            errorOutput: errorOutput.substring(Math.max(0, errorOutput.length - 500))
          });
        }
      }
    });

    pythonProcess.on("close", (code) => {
      try {
        if (processTimeout) clearTimeout(processTimeout);
        
        console.log(`[INFO] Python process closed with code: ${code}`);
        console.log(`[INFO] Output length: ${output.length}, Error output length: ${errorOutput.length}`);
        
        // Log first 500 chars of output and error for debugging
        if (output) console.log(`[DEBUG] First output: ${output.substring(0, 500)}`);
        if (errorOutput) console.log(`[DEBUG] Last error: ${errorOutput.substring(Math.max(0, errorOutput.length - 500))}`);
        
        // Check if we have valid JSON output
        if (!output || output.trim().length === 0) {
          console.error("[ERROR] No output from Python process");
          return res.status(500).json({
            status: "error",
            message: "No output from recognition process",
            details: errorOutput.substring(errorOutput.length - 1000) // Last 1000 chars of error
          });
        }

        // Clean output - remove non-JSON prefixes if any
        let cleanOutput = output.trim();
        
        // If output starts with non-JSON character, find the first {
        if (!cleanOutput.startsWith('{')) {
          const jsonStart = cleanOutput.indexOf('{');
          if (jsonStart > 0) {
            console.warn(`[WARN] Removing prefix before JSON: ${cleanOutput.substring(0, jsonStart)}`);
            cleanOutput = cleanOutput.substring(jsonStart);
          } else {
            console.error(`[ERROR] No JSON object found in output`);
            return res.status(500).json({
              status: "error",
              message: "Invalid output format from recognition process",
              details: cleanOutput.substring(0, 500)
            });
          }
        }

        // Parse JSON output từ Python
        let result;
        try {
          result = JSON.parse(cleanOutput);
        } catch (parseErr) {
          console.error(`[ERROR] JSON parse failed: ${parseErr.message}`);
          console.log(`[DEBUG] Raw output trying to parse: ${cleanOutput.substring(0, 1000)}`);
          return res.status(500).json({
            status: "error",
            message: "Invalid JSON output from recognition",
            details: `Parse error: ${parseErr.message}, First 200 chars: ${cleanOutput.substring(0, 200)}`
          });
        }
        
        if (result.status === "success") {
          console.log(`[SUCCESS] Food recognized: ${result.predicted_label}`);
          return res.status(200).json({
            status: "success",
            data: result
          });
        } else {
          console.error(`[ERROR] Recognition failed: ${result.error}`);
          return res.status(400).json({
            status: "error",
            message: result.error || "Unknown error during recognition"
          });
        }
      } catch (err) {
        console.error(`[ERROR] Unexpected error: ${err.message}`);
        res.status(500).json({
          status: "error",
          message: "Failed to process recognition result",
          details: err.message
        });
      }
    });

    pythonProcess.on("error", (err) => {
      console.error("❌ Error spawning Python process:", err);
      res.status(500).json({
        status: "error",
        message: "Failed to start recognition process"
      });
    });

  } catch (error) {
    console.error("❌ Error in recognizeFood:", error);
    res.status(500).json({
      status: "error",
      message: error.message || "Internal server error"
    });
  }
};
// Lấy tất cả món ăn
// controllers/foodController.js
const getAllFoods = async (req, res) => {
  try {
    const email = req.params.email;
    
    // Validate email parameter
    if (!email) {
      return res.status(400).json({
        status: "error",
        message: "Email parameter is required",
        data: [],
      });
    }
    
    const foods = await foodService.getAllFoods(email); // trả về array

    res.status(200).json({
      status: "success",
      data: Array.isArray(foods) ? foods : [], // ✅ luôn là array
    });
  } catch (error) {
    console.error("❌ Error in getAllFoods:", error);
    res.status(500).json({
      status: "error",
      message: error.message,
      data: [], // ✅ fallback array trống
    });
  }
};


module.exports = {
  searchFood,
  createFood,
  getAllFoods,
  getFoodsByDate,
  recognizeFood, // ✅ thêm endpoint nhận diện
};
