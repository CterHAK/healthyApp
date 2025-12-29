
const mongoose = require("mongoose");

const ExerciseGroupSchema = new mongoose.Schema({
  email: { type: String, required: true },
  group_name: { type: String, required: true },
}, { collection: "ExerciseGroup" });

module.exports = mongoose.models.ExerciseGroup || mongoose.model("ExerciseGroup", ExerciseGroupSchema);
