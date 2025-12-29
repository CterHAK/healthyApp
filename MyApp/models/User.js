// models/User.js
export default class User {
  constructor({
    email = "",
    password = "",
    gender = "—",
    name = "—",
    height = 170,
    weight = 60,
    age = 25,
    target = "—",
    targetWeight = 60,
    exercise = "—",
    allergies = [],
    diseases = [],
    caloriePlan = "—",
    bmr = 0,
    tdee = 0,
  } = {}) {
    this.email = email;
    this.password = password; // nếu cần lưu password tạm
    this.gender = gender;
    this.name = name;
    this.height = height;
    this.weight = weight;
    this.age = age;
    this.target = target;
    this.targetWeight = targetWeight;
    this.exercise = exercise;
    this.allergies = allergies;
    this.diseases = diseases;
    this.caloriePlan = caloriePlan;
    this.bmr = bmr;
    this.tdee = tdee;
  }

  // Hàm cập nhật từng trường
  updateField(key, value) {
    if (key in this) {
      this[key] = value;
    }
  }

  // Lấy data để gửi API
  toJSON() {
    const { password, ...data } = this; // loại password nếu không cần gửi
    return data;
  }
}
