const mongoose = require("mongoose");

const WorkoutPlanSchema = new mongoose.Schema({
  email: { type: String, required: true },
  group_name: { type: String, required: true },
  day: { type: String, required: true },
  session: { type: String, required: true },
  done_flag: { type: Boolean, default: false }
});

// Nếu model đã tồn tại, dùng model cũ
module.exports = mongoose.models.WorkoutPlan || mongoose.model("WorkoutPlan", WorkoutPlanSchema);
