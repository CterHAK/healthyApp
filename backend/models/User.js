const mongoose = require("mongoose");

const UserSchema = new mongoose.Schema({
  email: { type: String, required: true },
  gender: { type: String },
  height: { type: Number },
  weight: { type: Number },
  age: { type: Number },
  target: { type: String },
  targetWeight: { type: Number },
  exercise: { type: String, default: "—" },
  allergies: { type: [String], default: [] },
  diseases: { type: [String], default: [] },
  caloriePlan: { type: String, default: "—" },
  name: { type: String },
  bmr: { type: Number, default: 0 },
  tdee: { type: Number, default: 0 },
}, { collection: "Users" });


module.exports = mongoose.model("User", UserSchema);
