// models/ExerciseDetail.js
const mongoose = require("mongoose");

const ExerciseDetailSchema = new mongoose.Schema({
  email: { type: String, required: true },
  group_name: { type: String, required: true },
  name: { type: String, required: true },
  reps: { type: Number, default: 10 },
  sets: { type: Number, default: 3 },
}, { collection: "ExerciseDetail" });

module.exports = mongoose.models.ExerciseDetail || mongoose.model("ExerciseDetail", ExerciseDetailSchema);
