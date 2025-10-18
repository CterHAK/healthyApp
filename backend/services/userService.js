const User = require("../models/User");

const getAllUsers = async () => {
  return await User.find();
};

const createUser = async (data) => {
  const newUser = new User(data);
  return await newUser.save();
};

const getUserByEmail = async (email) => {
  return await User.findOne({ email });
};

module.exports = {
  getAllUsers,
  createUser,
  getUserByEmail
};
