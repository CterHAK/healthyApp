const userService = require("../services/userService");

const createUser = async (req, res) => {
  try {
    const user = await userService.createUser(req.body);
    res.status(201).json({ message: "User created", user });
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Lỗi tạo user", error: err.message });
  }
};

const getUsers = async (req, res) => {
  try {
    const users = await userService.getAllUsers();
    res.status(200).json(users);
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Lỗi lấy danh sách users", error: err.message });
  }
};

const getUserByEmail = async (req, res) => {
  try {
    const email = req.params.email;
    const user = await userService.getUserByEmail(email);
    if (!user) return res.status(404).json({ message: "Không tìm thấy user" });
    res.status(200).json(user);
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Lỗi lấy user theo email", error: err.message });
  }
};

module.exports = {
  createUser,
  getUsers,
  getUserByEmail,
};
