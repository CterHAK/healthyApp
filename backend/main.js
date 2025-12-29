const mongoose = require("mongoose");
if (mongoose.models.ExercisePlan) delete mongoose.models.ExercisePlan;
const ExercisePlan = require("./models/ExercisePlan");
