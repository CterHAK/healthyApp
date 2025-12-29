const Account = require("../models/Account");

// 🟢 Lấy tất cả tài khoản
const getAllAccounts = async () => {
  return await Account.find();
};

// 🟢 Tạo tài khoản mới
const createAccount = async (data) => {
  const newAccount = new Account(data);
  return await newAccount.save();
};

// 🟢 Tìm tài khoản theo email
const getAccountByEmail = async (email) => {
  return await Account.findOne({ email });
};

// 🟢 Cập nhật mật khẩu (nếu cần)
const updatePassword = async (email, newPassword) => {
  return await Account.findOneAndUpdate(
    { email },
    { password: newPassword },
    { new: true }
  );
};

// 🟢 Đăng nhập (kiểm tra email + password)
const loginAccount = async (email, password) => {
  const account = await Account.findOne({ email });
  if (!account) return null; // không tìm thấy email

  if (account.password !== password) return false; // sai mật khẩu

  return account; // đăng nhập thành công
};

module.exports = {
  getAllAccounts,
  createAccount,
  getAccountByEmail,
  updatePassword,
  loginAccount, // ✅ thêm dòng này
};
