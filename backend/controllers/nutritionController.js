const exerciseService = require("../services/exerciseService");
const nutritionService = require("../services/nutritionService");
const Food = require("../models/FoodRecognizerModel");
const WorkoutPlan = require("../models/WorkoutPlan");
const ExerciseDetail = require("../models/ExerciseDetail");

// ==================== NUTRITION ENDPOINTS ====================

/**
 * 📊 GET /api/exercises/daily-nutrition?email=xxx&date=2025-11-28&weight=70
 * Lấy tổng intake + burned calo cho 1 ngày
 */
async function getDailyNutrition(req, res) {
  try {
    const { email, date, weight = 70 } = req.query;

    if (!email || !date) {
      return res.status(400).json({
        status: "error",
        message: "Thiếu email hoặc date"
      });
    }

    const result = await nutritionService.getDailyNutrition(
      email,
      date,
      parseFloat(weight)
    );

    res.json(result);
  } catch (error) {
    console.error("❌ Error in getDailyNutrition:", error);
    res.status(500).json({
      status: "error",
      message: error.message
    });
  }
}

/**
 * 📈 GET /api/exercises/nutrition-range?email=xxx&days=30&weight=70
 * Lấy dữ liệu nutrition cho 30 ngày (hoặc 7 ngày)
 */
async function getDailyNutritionRange(req, res) {
  try {
    const { email, days = 30, weight = 70 } = req.query;

    if (!email) {
      return res.status(400).json({
        status: "error",
        message: "Thiếu email"
      });
    }

    const result = await nutritionService.getDailyNutritionRange(
      email,
      parseInt(days),
      parseFloat(weight)
    );

    res.json({
      status: "success",
      period_days: parseInt(days),
      data: result
    });
  } catch (error) {
    console.error("❌ Error in getDailyNutritionRange:", error);
    res.status(500).json({
      status: "error",
      message: error.message
    });
  }
}

/**
 * 📊 GET /api/exercises/nutrition-stats?email=xxx&days=30&weight=70
 * Lấy thống kê (avg, total, min, max)
 */
async function getNutritionStats(req, res) {
  try {
    const { email, days = 30, weight = 70 } = req.query;

    if (!email) {
      return res.status(400).json({
        status: "error",
        message: "Thiếu email"
      });
    }

    const result = await nutritionService.getNutritionStats(
      email,
      parseInt(days),
      parseFloat(weight)
    );

    res.json(result);
  } catch (error) {
    console.error("❌ Error in getNutritionStats:", error);
    res.status(500).json({
      status: "error",
      message: error.message
    });
  }
}

/**
 * 📉 GET /api/exercises/chart-data?email=xxx&days=30&weight=70
 * Lấy data đã format cho biểu đồ
 */
async function getChartData(req, res) {
  try {
    const { email, days = 30, weight = 70 } = req.query;

    if (!email) {
      return res.status(400).json({
        status: "error",
        message: "Thiếu email"
      });
    }

    const result = await nutritionService.getChartData(
      email,
      parseInt(days),
      parseFloat(weight)
    );

    res.json(result);
  } catch (error) {
    console.error("❌ Error in getChartData:", error);
    res.status(500).json({
      status: "error",
      message: error.message
    });
  }
}

/**
 *  DEBUG: Lấy tất cả food records của user
 * GET /api/exercises/debug-foods?email=xxx
 */
async function debugFoods(req, res) {
  try {
    const { email } = req.query;

    if (!email) {
      return res.status(400).json({
        status: "error",
        message: "Thiếu email"
      });
    }

    // Lấy tất cả food records
    const foods = await Food.find({ email }).limit(100).sort({ createdAt: -1 }).lean();

    res.json({
      status: "success",
      total: foods.length,
      records: foods.map(f => ({
        _id: f._id,
        day: f.day,
        seassion: f.seassion,
        dish_name: f.dish_name,
        calories: f.nutrition?.calories_kcal || 0,
        createdAt: f.createdAt
      }))
    });
  } catch (error) {
    console.error("❌ Error in debugFoods:", error);
    res.status(500).json({
      status: "error",
      message: error.message
    });
  }
}

/**
 * 🔍 DEBUG: Lấy tất cả exercises (WorkoutPlan + ExerciseDetail)
 * GET /api/exercises/debug-exercises?email=xxx&date=2025-11-28
 */
async function debugExercises(req, res) {
  try {
    const { email, date } = req.query;

    if (!email || !date) {
      return res.status(400).json({
        status: "error",
        message: "Thiếu email hoặc date"
      });
    }

    // Lấy tất cả WorkoutPlan (completed và incomplete)
    const allPlans = await WorkoutPlan.find({ email, day: date }).lean();
    const completedPlans = await WorkoutPlan.find({ email, day: date, done_flag: true }).lean();
    
    // Lấy tất cả ExerciseDetail
    const details = await ExerciseDetail.find({ email }).limit(100).sort({ createdAt: -1 }).lean();

    res.json({
      status: "success",
      date,
      summary: {
        totalPlans: allPlans.length,
        completedPlans: completedPlans.length,
        exerciseDetailsCount: details.length
      },
      workoutPlans: {
        all: allPlans.map(p => ({
          _id: p._id,
          day: p.day,
          group_name: p.group_name,
          session: p.session,
          done_flag: p.done_flag,
          calories_burned: p.calories_burned || "NOT_SET"
        })),
        completed: completedPlans.map(p => ({
          _id: p._id,
          group_name: p.group_name,
          session: p.session,
          calories_burned: p.calories_burned || "NOT_SET"
        }))
      },
      exerciseDetails: {
        total: details.length,
        records: details.map(d => ({
          _id: d._id,
          group_name: d.group_name,
          name: d.name,
          sets: d.sets,
          reps: d.reps
        }))
      }
    });
  } catch (error) {
    console.error("❌ Error in debugExercises:", error);
    res.status(500).json({
      status: "error",
      message: error.message
    });
  }
}

/**
 * 🔍 DEBUG: Chi tiết ngày hôm nay
 * GET /api/exercises/debug-today?email=xxx&date=2025-11-28
 */
async function debugToday(req, res) {
  try {
    const { email, date = new Date().toISOString().split("T")[0] } = req.query;

    if (!email) {
      return res.status(400).json({
        status: "error",
        message: "Thiếu email"
      });
    }

    // Lấy tất cả foods của ngày
    const foods = await Food.find({ email, day: date }).lean();
    
    // Lấy tất cả WorkoutPlan của ngày
    const allPlans = await WorkoutPlan.find({ email, day: date }).lean();
    const completedPlans = await WorkoutPlan.find({ email, day: date, done_flag: true }).lean();

    // Lấy 5 food records gần đây để xem định dạng day
    const recentFoods = await Food.find({ email }).sort({ createdAt: -1 }).limit(5).lean();

    res.json({
      status: "success",
      queryDate: date,
      serverDate: new Date().toISOString(),
      results: {
        foods: {
          count: foods.length,
          records: foods.map(f => ({
            day: f.day,
            seassion: f.seassion,
            dish_name: f.dish_name,
            calories: f.nutrition?.calories_kcal || 0
          }))
        },
        workoutPlans: {
          total: allPlans.length,
          completed: completedPlans.length,
          records: allPlans.map(p => ({
            day: p.day,
            group_name: p.group_name,
            session: p.session,
            done_flag: p.done_flag,
            calories_burned: p.calories_burned || "NOT_SET"
          }))
        },
        recentFoodsFormat: {
          count: recentFoods.length,
          samples: recentFoods.map(f => ({
            day: f.day,
            createdAt: f.createdAt
          }))
        }
      }
    });
  } catch (error) {
    console.error("❌ Error in debugToday:", error);
    res.status(500).json({
      status: "error",
      message: error.message
    });
  }
}

module.exports = {
  getDailyNutrition,
  getDailyNutritionRange,
  getNutritionStats,
  getChartData,
  debugFoods,
  debugExercises,
  debugToday
};
