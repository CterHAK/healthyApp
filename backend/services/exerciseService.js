const ExerciseDetail = require("../models/ExerciseDetail");
const WorkoutPlan = require("../models/WorkoutPlan");
const ExerciseGroup = require("../models/ExerciseGroup");
const { runPython } = require("../services/pythonService");

// --------------------- Group ---------------------
async function addExerciseGroup(data) {
  // data: { email, group_name }
  const group = new ExerciseGroup(data);
  return await group.save();
}

async function getGroup(email, group_name) {
  return await ExerciseGroup.findOne({ email, group_name }).lean();
}

async function getGroupsByEmail(email) {
  return await ExerciseGroup.find({ email }).lean();
}

// --------------------- ExerciseDetail ---------------------
async function addExerciseDetail(data) {
  // data: { email, group_name, name, sets, reps }
  const detail = new ExerciseDetail(data);
  return await detail.save();
}

async function getExerciseDetail(email, group_name, name) {
  return await ExerciseDetail.findOne({ email, group_name, name });
}

async function getExercisesByEmailAndGroup(email, group_name) {
  return await ExerciseDetail.find({ email, group_name }).lean();
}

// --------------------- Plan ---------------------
async function addExercisePlan(data) {
  // data: { email, group_name, day, session, done_flag }
  const plan = new WorkoutPlan(data);
  return await plan.save();
}

async function getPlansByEmail(email) {
  return await WorkoutPlan.find({ email }).lean();
}

async function getPlansByEmailAndDay(email, day) {
  return await WorkoutPlan.find({ email, day }).lean();
}

// --------------------- Python API ---------------------
async function filterExercises(filters) {
  return await runPython({ action: "filter", ...filters });
}

async function getExerciseInfoByName(name) {
  return await runPython({ action: "get_exercise_info", exercise_name: name });
}

async function estimateCaloriesByName({ exercise_name, sets, reps, weight_kg }) {
  return await runPython({ 
    action: "calories_name_only", 
    exercise_name, 
    sets, 
    reps, 
    weight_kg 
  });
}

async function getMuscles() {
  return await runPython({ action: "muscles" });
}

async function getEquipment() {
  return await runPython({ action: "equipment" });
}

async function getDependentFilters(filters) {
  return await runPython({ action: "dependent_filters", ...filters });
}


async function updateDoneFlag(email, group_name, day, session, weight_kg = 70) {
  // Lấy ExerciseDetail của group này để tính calories
  const details = await ExerciseDetail.find({ email, group_name }).lean();
  
  let totalCalories = 0;
  
  // Tính tổng calories từ tất cả exercises trong group
  for (const detail of details) {
    const caloResult = await estimateCaloriesByName({
      exercise_name: detail.name,
      sets: detail.sets,
      reps: detail.reps,
      weight_kg
    });
    
    if (caloResult && caloResult.calories) {
      totalCalories += caloResult.calories;
    }
  }
  
  console.log(`[DEBUG] updateDoneFlag: group=${group_name}, exercises=${details.length}, calories=${totalCalories}`);
  
  // Update WorkoutPlan với done_flag = true và calories_burned
  return await WorkoutPlan.findOneAndUpdate(
    { email, group_name, day, session },
    { 
      done_flag: true,
      calories_burned: totalCalories  // ⭐ Lưu calories khi mark done
    },
    { new: true } // trả về document sau khi update
  );
}

module.exports = {
  addExerciseGroup,
  getGroup,
  getGroupsByEmail,
  addExerciseDetail,
  getExerciseDetail,
  getExercisesByEmailAndGroup,
  addExercisePlan,
  getPlansByEmail,
  getPlansByEmailAndDay,
  filterExercises,
  getExerciseInfoByName,
  estimateCaloriesByName,
  getMuscles,
  getEquipment,
  getDependentFilters,
  updateDoneFlag,
};
