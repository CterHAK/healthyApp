const Food = require("../models/FoodRecognizerModel");
const WorkoutPlan = require("../models/WorkoutPlan");
const ExerciseDetail = require("../models/ExerciseDetail");
const { runPython } = require("./pythonService");
const exerciseService = require("./exerciseService");

// ==================== DAILY NUTRITION ====================

/**
 * 📊 Lấy tổng intake + burned calo cho 1 ngày
 * @param {string} email
 * @param {string} date - Format: "2025-11-28"
 * @param {number} weight_kg - Cân nặng hiện tại (để tính burned)
 * @returns {Object} { date, intake, burned, net, details }
 */
async function getDailyNutrition(email, date, weight_kg = 70) {
  try {
    // 1️⃣ Lấy intake từ Food collection
    // Hỗ trợ nhiều định dạng ngày: "2025-11-28", "28-11-2025", "28/11/2025", "Thứ X"
    
    // Chuyển đổi ngày nhập vào sang các định dạng khác
    const inputDate = new Date(date + "T00:00:00Z"); // Parse as UTC to avoid timezone issues
    const todayIso = date; // Sử dụng date param trực tiếp (đã là ISO format từ frontend)
    const todayDMY = inputDate.toLocaleDateString("vi-VN"); // "28/11/2025"
    const todayDash = `${inputDate.getUTCDate()}-${inputDate.getUTCMonth() + 1}-${inputDate.getUTCFullYear()}`; // "28-11-2025"
    
    const foods = await Food.find({ 
      email,
      $or: [
        { day: date },           // Original format từ frontend
        { day: todayIso },       // "2025-11-28"
        { day: todayDMY },       // "28/11/2025"
        { day: todayDash }       // "28-11-2025"
      ]
    }).lean();

    console.log(`[DEBUG] Query Foods: email=${email}, date=${date}, ISO=${todayIso}, DMY=${todayDMY}, Dash=${todayDash}, found=${foods.length}`);
    
    let intake = {
      total_calories: 0,
      protein: 0,
      carb: 0,
      fat: 0,
      meals: []
    };

    foods.forEach((food) => {
      const nutrition = food.nutrition || {};
      intake.total_calories += nutrition.calories_kcal || 0;
      intake.protein += nutrition.protein_g || 0;
      intake.carb += nutrition.carbohydrate_g || 0;
      intake.fat += nutrition.fat_g || 0;
      
      intake.meals.push({
        session: food.seassion,
        name: food.dish_name,
        calories: nutrition.calories_kcal || 0
      });
    });

    console.log(`[DEBUG] Intake Total: ${intake.total_calories} kcal, Meals: ${intake.meals.length}`);

    // 2️⃣ Lấy burned từ WorkoutPlan + tính calo từ ExerciseDetail
    // ⚠️ CHỈ lấy WorkoutPlan của ngày hôm nay và có done_flag = true (đã hoàn thành)
    const plans = await WorkoutPlan.find({ 
      email, 
      day: date,
      done_flag: true  // ⭐ CHỈ tính calo từ buổi tập đã hoàn thành
    }).lean();
    
    let burned = {
      total_calories: 0,
      exercises: []
    };

    console.log(`[DEBUG] Found COMPLETED WorkoutPlans for day ${date}: ${plans.length}`);

    for (const plan of plans) {
      // Nếu đã lưu calo trước đó (nên có vì tính lúc mark done)
      if (plan.calories_burned && plan.calories_burned > 0) {
        burned.total_calories += plan.calories_burned;
        burned.exercises.push({
          session: plan.session,
          group_name: plan.group_name,
          calories: plan.calories_burned
        });
        console.log(`[DEBUG] ✓ Using saved calories for ${plan.group_name}: ${plan.calories_burned} kcal`);
      } else {
        // Fallback: Tính lại từ ExerciseDetail nếu chưa có lưu
        console.log(`[DEBUG] ⚠️ No saved calories, recalculating for ${plan.group_name}`);
        
        const details = await ExerciseDetail.find({
          email,
          group_name: plan.group_name
        }).lean();

        console.log(`[DEBUG] Found ExerciseDetails for group ${plan.group_name}: ${details.length}`);

        let groupCalories = 0;
        for (const detail of details) {
          const caloResult = await exerciseService.estimateCaloriesByName({
            exercise_name: detail.name,
            sets: detail.sets,
            reps: detail.reps,
            weight_kg
          });

          console.log(`[DEBUG] Exercise ${detail.name}: sets=${detail.sets}, reps=${detail.reps}, calories=${caloResult?.calories || 0}`);

          if (caloResult && caloResult.calories) {
            groupCalories += caloResult.calories;
          }
        }

        burned.total_calories += groupCalories;
        burned.exercises.push({
          session: plan.session,
          group_name: plan.group_name,
          calories: groupCalories
        });
        console.log(`[DEBUG] ✓ Calculated ${groupCalories} kcal for ${plan.group_name}`);
      }
    }

    console.log(`[DEBUG] FINAL Total Burned: ${burned.total_calories} kcal from ${burned.exercises.length} exercise(s)`);

    // 3️⃣ Tính net
    const net_calories = intake.total_calories - burned.total_calories;

    return {
      date,
      intake,
      burned,
      net_calories,
      status: "success"
    };
  } catch (error) {
    console.error("❌ Error in getDailyNutrition:", error);
    return {
      status: "error",
      message: error.message,
      date,
      intake: { total_calories: 0, protein: 0, carb: 0, fat: 0, meals: [] },
      burned: { total_calories: 0, exercises: [] },
      net_calories: 0
    };
  }
}

/**
 * 📈 Lấy dữ liệu nutrition cho nhiều ngày (7 hoặc 30 ngày)
 * @param {string} email
 * @param {number} days - 7 hoặc 30 ngày
 * @param {number} weight_kg
 * @returns {Array} Mảng các ngày với intake, burned, net
 */
async function getDailyNutritionRange(email, days = 30, weight_kg = 70) {
  try {
    const result = [];
    
    // Tạo mảng các ngày (đơn giản, dùng getDate/getMonth/getFullYear)
    for (let i = days - 1; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, "0");
      const day = String(date.getDate()).padStart(2, "0");
      const dateStr = `${year}-${month}-${day}`;
      
      console.log(`[DEBUG] getDailyNutritionRange: i=${i}, dateStr=${dateStr}`);
      
      const nutrition = await getDailyNutrition(email, dateStr, weight_kg);
      result.push(nutrition);
    }

    return result;
  } catch (error) {
    console.error("❌ Error in getDailyNutritionRange:", error);
    return [];
  }
}

/**
 * 📊 Lấy stats tổng hợp cho khoảng thời gian
 * @param {string} email
 * @param {number} days
 * @param {number} weight_kg
 * @returns {Object} Thống kê trung bình, tổng, min, max
 */
async function getNutritionStats(email, days = 30, weight_kg = 70) {
  try {
    const dailyData = await getDailyNutritionRange(email, days, weight_kg);
    
    // Lọc các ngày có dữ liệu
    const validDays = dailyData.filter(d => d.status === "success");
    
    if (validDays.length === 0) {
      return {
        status: "error",
        message: "Không có dữ liệu",
        period_days: days
      };
    }

    // Tính toán
    const intakes = validDays.map(d => d.intake.total_calories);
    const burneds = validDays.map(d => d.burned.total_calories);
    const nets = validDays.map(d => d.net_calories);

    const calculateStats = (arr) => ({
      avg: (arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(0),
      total: arr.reduce((a, b) => a + b, 0).toFixed(0),
      max: Math.max(...arr).toFixed(0),
      min: Math.min(...arr).toFixed(0)
    });

    return {
      status: "success",
      period_days: days,
      intake_stats: calculateStats(intakes),
      burned_stats: calculateStats(burneds),
      net_stats: calculateStats(nets),
      data: validDays
    };
  } catch (error) {
    console.error("❌ Error in getNutritionStats:", error);
    return {
      status: "error",
      message: error.message
    };
  }
}

/**
 * 📉 Lấy data cho biểu đồ (đã format sẵn)
 * @param {string} email
 * @param {number} days
 * @param {number} weight_kg
 * @returns {Object} { labels, intakeData, burnedData }
 */
async function getChartData(email, days = 30, weight_kg = 70) {
  try {
    const dailyData = await getDailyNutritionRange(email, days, weight_kg);
    
    const labels = [];
    const intakeData = [];
    const burnedData = [];

    dailyData.forEach((day) => {
      const date = new Date(day.date);
      labels.push(`${date.getDate()}/${date.getMonth() + 1}`);
      intakeData.push(day.intake.total_calories || 0);
      burnedData.push(day.burned.total_calories || 0);
    });

    return {
      status: "success",
      labels,
      datasets: [
        {
          label: "Intake",
          data: intakeData,
          borderColor: "#FF6B6B",
          backgroundColor: "rgba(255, 107, 107, 0.1)",
          fill: true
        },
        {
          label: "Burned",
          data: burnedData,
          borderColor: "#4ECDC4",
          backgroundColor: "rgba(78, 205, 196, 0.1)",
          fill: true
        }
      ]
    };
  } catch (error) {
    console.error("❌ Error in getChartData:", error);
    return {
      status: "error",
      message: error.message
    };
  }
}

module.exports = {
  getDailyNutrition,
  getDailyNutritionRange,
  getNutritionStats,
  getChartData
};
