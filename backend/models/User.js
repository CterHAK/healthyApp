const mongoose = require("mongoose");

const UserSchema = new mongoose.Schema({
  email: { type: String, required: true },
  gender: { type: String },
  height: { type: Number },
  weight: { type: Number },
  age: { type: Number },
  target: { type: String },
  targetWeight: { type: Number },
  exercise: { type: String },
  allergy: { type: String },
  disease: { type: String },
  caloriePlan: { type: String },
  name: { type: String }
}, { collection: "Users" });

module.exports = mongoose.model("User", UserSchema);
