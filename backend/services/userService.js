const User = require("../models/User");

const createUser = async (data) => {
  // đảm bảo allergies/diseases luôn là array, bmr/tdee là number
  const userData = {
    ...data,
    allergies: Array.isArray(data.allergies) ? data.allergies : [],
    diseases: Array.isArray(data.diseases) ? data.diseases : [],
    bmr: Number(data.bmr) || 0,
    tdee: Number(data.tdee) || 0,
  };
  const user = new User(userData);
  return await user.save();
};

const getAllUsers = async () => {
  const users = await User.find({});
  return users.map(u => ({
    ...u.toObject(),
    allergies: Array.isArray(u.allergies) ? u.allergies : [],
    diseases: Array.isArray(u.diseases) ? u.diseases : [],
  }));
};

const getUserByEmail = async (email) => {
  const user = await User.findOne({ email });
  if (!user) return null;
  return {
    ...user.toObject(),
    allergies: Array.isArray(user.allergies) ? user.allergies : [],
    diseases: Array.isArray(user.diseases) ? user.diseases : [],
  };
};

module.exports = {
  createUser,
  getAllUsers,
  getUserByEmail,
};
