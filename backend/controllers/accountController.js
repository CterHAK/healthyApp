const accountService = require("../services/accountService");

const getAccounts = async (req, res) => {
  try {
    const accounts = await accountService.getAllAccounts();
    res.json(accounts);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

const addAccount = async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password)
      return res.status(400).json({ message: "Email và mật khẩu là bắt buộc" });

    const existingAccount = await accountService.getAccountByEmail(email);
    if (existingAccount)
      return res.status(400).json({ message: "Email đã tồn tại" });

    const newAccount = await accountService.createAccount({ email, password });
    res.status(201).json({ message: "Tạo tài khoản thành công", account: newAccount });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// 🟢 Đăng nhập
const loginAccount = async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password)
      return res.status(400).json({ message: "Email và mật khẩu là bắt buộc" });

    const account = await accountService.getAccountByEmail(email);
    if (!account) return res.status(404).json({ message: "Email không tồn tại" });

    if (account.password !== password)
      return res.status(401).json({ message: "Sai mật khẩu" });

    res.status(200).json({ message: "Đăng nhập thành công", account });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

const getAccountByEmail = async (req, res) => {
  try {
    const { email } = req.params;
    const account = await accountService.getAccountByEmail(email);
    if (!account) return res.status(404).json({ message: "Không tìm thấy tài khoản" });
    res.json(account);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

const updatePassword = async (req, res) => {
  try {
    const { email } = req.params;
    const { password } = req.body;
    if (!password)
      return res.status(400).json({ message: "Mật khẩu mới là bắt buộc" });

    const updated = await accountService.updatePassword(email, password);
    if (!updated)
      return res.status(404).json({ message: "Không tìm thấy tài khoản" });

    res.json({ message: "Đổi mật khẩu thành công" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

module.exports = {
  getAccounts,
  addAccount,
  loginAccount, // ✅ thêm dòng này
  getAccountByEmail,
  updatePassword,
};
