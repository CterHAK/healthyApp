const express = require("express");
const router = express.Router();
const exerciseController = require("../controllers/exerciseController");
const nutritionController = require("../controllers/nutritionController");

// -------------------- Exercise --------------------

// Lọc bài tập
router.get("/filter", exerciseController.filterExercises);

// Lấy dependent filters
router.get("/dependent-filters", exerciseController.getDependentFilters);

// Danh sách cơ
router.get("/muscles", exerciseController.getMuscles);

// Danh sách thiết bị
router.get("/equipment", exerciseController.getEquipment);

// Tra cứu thông tin bài tập
router.get("/exercise-info", exerciseController.getExerciseInfo);

// Tính calories
router.post("/calculate-calories", exerciseController.calculateCalories);

// Thêm nhiều bài tập vào nhóm
router.post("/group", exerciseController.addExerciseGroupAndDetail);

// Lấy nhóm + chi tiết bài tập
router.get("/group", exerciseController.getExerciseGroup);

// -------------------- Plan --------------------

// Thêm vào plan
router.post("/plan", exerciseController.addExercisePlan);

// Lấy tất cả plan theo email
router.get("/plan", exerciseController.getExercisePlans);

router.put("/plan/done", exerciseController.markWorkoutDone);

// -------------------- Nutrition Analytics --------------------

// Lấy calo hôm nay
router.get("/daily-nutrition", nutritionController.getDailyNutrition);

// Lấy calo khoảng thời gian (7 hoặc 30 ngày)
router.get("/nutrition-range", nutritionController.getDailyNutritionRange);

// Lấy thống kê calo
router.get("/nutrition-stats", nutritionController.getNutritionStats);

// Lấy data cho biểu đồ
router.get("/chart-data", nutritionController.getChartData);

//  DEBUG: Lấy tất cả food records
router.get("/debug-foods", nutritionController.debugFoods);

// 🔍 DEBUG: Lấy tất cả exercises (WorkoutPlan + ExerciseDetail)
router.get("/debug-exercises", nutritionController.debugExercises);

// 🔍 DEBUG: Chi tiết ngày hôm nay
router.get("/debug-today", nutritionController.debugToday);

module.exports = router;
