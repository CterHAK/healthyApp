const userService = require("../services/userService");

const getUsers = async (req, res) => {
  try {
    const users = await userService.getAllUsers();
    res.json(users);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

const addUser = async (req, res) => {
  try {
    const newUser = await userService.createUser(req.body);
    res.status(201).json({ message: "User created", user: newUser });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};
const getUserByEmail = async (req, res) => {
  try {
    const { email } = req.params; // email từ URL
    const user = await userService.getUserByEmail(email);
    if (!user) return res.status(404).json({ message: "User not found" });
    res.json(user);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

module.exports = { getUsers, addUser, getUserByEmail };
