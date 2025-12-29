const mongoose = require("mongoose");

const AccountSchema = new mongoose.Schema({
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true }
}, { collection: "Accounts" });

module.exports = mongoose.model("Account", AccountSchema);
