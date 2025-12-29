import React, { useState } from "react";
import { View, Text, TextInput, TouchableOpacity, Image, Alert, StyleSheet } from "react-native";

import ProgressBar from "../components/ProgressBar.js";
import Button from "../components/Button.js";
import HeightAdjustTab from "../components/HeightAdjustTab.js";
import WeightAdjustTab from "../components/WeightAdjustTab.js";
import AgeTab from "../components/AgeTab.js";
import TargetTab from "../components/TargetTab.js";
import TargetWeightBMISlider from "../components/TargetWeightTab.js";
import ExerciseTab from "../components/ExerciseTab.js";
import AllergyTab from "../components/AllergyTab.js";
import DiseaseTab from "../components/DiseaseTab.js";
import CaloriePlanTab from "../components/CaloriePlanTab.js";

import { registerAccount } from "../services/account_api.js";
import { createUser } from "../services/user_api.js";

import User from "../models/User.js";

const steps = [
  { key: "account", label: "Thông tin tài khoản", type: "account" },
  { key: "gender", label: "Giới tính", type: "radio", options: ["Nam", "Nữ"] },
  { key: "height", label: "Chiều cao", type: "height" },
  { key: "weight", label: "Cân nặng", type: "weight" },
  { key: "age", label: "Tuổi", type: "age" },
  { key: "target", label: "Mục tiêu", type: "target" },
  { key: "targetWeight", label: "Cân nặng mục tiêu", type: "targetWeight" },
  { key: "exercise", label: "Mức độ vận động", type: "exercise" },
  { key: "allergy", label: "Dị ứng thực phẩm", type: "allergy" },
  { key: "disease", label: "Bệnh nền", type: "disease" },
  { key: "caloriePlan", label: "Kế hoạch calo", type: "caloriePlan" },
  { key: "name", label: "Tên", type: "text", placeholder: "Nhập tên của bạn" },
];

const genderImages = {
  Nam: require("../assets/male.png"),
  Nữ: require("../assets/female.png"),
};

export default function RegisterScreen({ navigation }) {
  const [step, setStep] = useState(0);
  const [loading, setLoading] = useState(false);

  // Khởi tạo User class với confirmPassword
  const [user, setUser] = useState(new User({ confirmPassword: "" }));

  const updateUserField = (key, value) => {
    const updatedUser = { ...user };
    updatedUser[key] = value;
    setUser(updatedUser);
  };

  const handleNext = () => {
    const current = steps[step];

    if (current.type === "account") {
      if (!user.email || !user.password || !user.confirmPassword) {
        Alert.alert("Thiếu thông tin", "Vui lòng nhập đầy đủ email và mật khẩu.");
        return;
      }
      if (user.password !== user.confirmPassword) {
        Alert.alert("Lỗi", "Mật khẩu xác nhận không khớp.");
        return;
      }
    }

    if (step < steps.length - 1) setStep(step + 1);
    else handleRegister();
  };

  const handlePrevious = () => {
    if (step > 0) setStep(step - 1);
    else if (navigation) navigation.goBack();
  };

  const handleRegister = async () => {
    try {
      setLoading(true);

      // 1️⃣ Tạo account
      const accountRes = await registerAccount(user.email, user.password);
      console.log("✅ Account created:", accountRes);

      // 2️⃣ Gửi data user (loại password)
      const userData = { ...user };
      delete userData.confirmPassword;
      const userRes = await createUser(userData);
      console.log("✅ User created:", userRes);

      Alert.alert("Thành công", "Tài khoản và hồ sơ cá nhân đã được tạo!");
      navigation.replace("Home", { userData });
    } catch (error) {
      console.error("❌ Lỗi đăng ký:", error);
      Alert.alert("Lỗi", error.response?.data?.message || "Đăng ký thất bại.");
    } finally {
      setLoading(false);
    }
  };

  const renderInput = () => {
    const current = steps[step];

    switch (current.type) {
      case "account":
        return (
          <View>
            <TextInput
              placeholder="Nhập email"
              value={user.email}
              onChangeText={(text) => updateUserField("email", text)}
            />
            <TextInput
              placeholder="Nhập mật khẩu"
              secureTextEntry
              value={user.password}
              onChangeText={(text) => updateUserField("password", text)}
            />
            <TextInput
              placeholder="Xác nhận mật khẩu"
              secureTextEntry
              value={user.confirmPassword}
              onChangeText={(text) => updateUserField("confirmPassword", text)}
            />
          </View>
        );

      case "text":
        return (
          <TextInput
            placeholder={current.placeholder}
            value={user[current.key]}
            onChangeText={(text) => updateUserField(current.key, text)}
          />
        );

      case "radio":
        return (
          <View style={styles.genderRow}>
            {current.options.map((option) => (
              <TouchableOpacity
                key={option}
                style={[
                  styles.genderOption,
                  user.gender === option && styles.genderOptionActive,
                ]}
                onPress={() => updateUserField("gender", option)}
              >
                <Image source={genderImages[option]} style={styles.genderImage} />
                <Text style={styles.genderText}>{option}</Text>
              </TouchableOpacity>
            ))}
          </View>
        );

      case "height":
        return (
          <HeightAdjustTab
            source={user.gender ? genderImages[user.gender] : genderImages["Nam"]}
            height={user.height}
            onHeightChange={(h) => updateUserField("height", h)}
          />
        );

      case "weight":
        return (
          <WeightAdjustTab
            height={user.height}
            weight={user.weight}
            onWeightChange={(w) => updateUserField("weight", w)}
          />
        );

      case "age":
        return (
          <AgeTab
            min={1}
            max={100}
            value={user.age}
            onSelect={(a) => updateUserField("age", a)}
          />
        );

      case "target":
        return (
          <TargetTab
            selectedTarget={user.target}
            onSelect={(t) => updateUserField("target", t)}
          />
        );

      case "targetWeight":
        return (
          <TargetWeightBMISlider
            height={user.height}
            currentWeight={user.weight}
            targetWeight={user.targetWeight}
            onTargetWeightChange={(w) => updateUserField("targetWeight", w)}
            selectedTarget={user.target}
          />
        );

      case "exercise":
        return (
          <ExerciseTab
            gender={user.gender === "Nam" ? "male" : "female"}
            weight={user.weight}
            height={user.height}
            age={user.age}
            onResult={(result) => {
              setUser(prev => ({
                ...prev,
                exercise: result.exercise,
                bmr: result.bmr,
                tdee: result.tdee,
              }));
            }}
          />

        );

      case "allergy":
        return (
          <AllergyTab
            selectedFoods={user.allergies}
            onChange={(list, other) =>
              updateUserField("allergies", [...list, other].filter(Boolean))
            }
          />
        );

      case "disease":
        return (
          <DiseaseTab
            selectedDiseases={user.diseases}
            onChange={(list, other) =>
              updateUserField("diseases", [...list, other].filter(Boolean))
            }
          />
        );

      case "caloriePlan":
        return (
          <CaloriePlanTab
            weight={user.weight}
            targetWeight={user.targetWeight}
            tdee={user.tdee}
            onChange={(plan) => updateUserField("caloriePlan", plan)}
          />

        );

      default:
        return null;
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.progressWrapper}>
        <ProgressBar steps={steps} step={step} />
      </View>

      <View style={styles.content}>
        <Text style={styles.label}>{steps[step].label}</Text>
        {renderInput()}
      </View>

      <View style={styles.buttonRow}>
        <Button title={step > 0 ? "Quay lại" : "Hủy"} onPress={handlePrevious} />
        <Button
          title={loading ? "Đang xử lý..." : step < steps.length - 1 ? "Tiếp tục" : "Hoàn tất"}
          onPress={handleNext}
        />
      </View>
    </View>
  );
}

// Giữ nguyên style cũ
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#fff" },
  progressWrapper: {
    position: "absolute",
    top: 100,
    left: 0,
    right: 0,
    alignItems: "center",
    zIndex: 10,
  },
  content: {
    flex: 1,
    marginTop: 120,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 20,
  },
  label: { fontSize: 18, marginBottom: 10 },
  input: {
    width: "80%",
    borderWidth: 1,
    borderColor: "#ccc",
    padding: 10,
    marginBottom: 15,
    borderRadius: 5,
  },
  buttonRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    padding: 20,
    position: "absolute",
    bottom: 30,
    left: 0,
    right: 0,
  },
  genderRow: {
    flexDirection: "row",
    justifyContent: "center",
    marginBottom: 20,
  },
  genderOption: {
    alignItems: "center",
    marginHorizontal: 20,
    borderWidth: 2,
    borderColor: "#ccc",
    borderRadius: 12,
    padding: 8,
  },
  genderOptionActive: {
    borderColor: "#e53935",
    backgroundColor: "#fff0f0",
  },
  genderImage: { width: 120, height: 200 },
  genderText: { fontSize: 16 },
});
