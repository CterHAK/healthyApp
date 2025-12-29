import React from "react";
import { View, Text, StyleSheet } from "react-native";
import Slider from "@react-native-community/slider";

export default function TargetWeightBMISlider({
  height,
  currentWeight,
  targetWeight,
  onTargetWeightChange,
  selectedTarget,
}) {
  const minWeight = 30;
  const maxWeight = 200;
  const step = 1;

  const calculateBMI = (weightValue) => {
    const heightInMeters = height / 100;
    const bmi = weightValue / (heightInMeters * heightInMeters);
    return bmi.toFixed(1);
  };

  const getBMILevel = (bmi) => {
    if (bmi < 18.5) return "Thiếu cân";
    if (bmi < 25) return "Bình thường";
    if (bmi < 30) return "Thừa cân";
    return "Béo phì";
  };

  const bmiValue = calculateBMI(targetWeight);
  const bmiLevel = getBMILevel(bmiValue);

  // 🔎 Kiểm tra xung đột mục tiêu
  let warning = "";
  if (selectedTarget === "lose_weight" && targetWeight > currentWeight) {
    warning = "⚠️ Bạn đang muốn giảm cân nhưng cân nặng mục tiêu lại cao hơn hiện tại!";
  }
  if (selectedTarget === "gain_weight" && targetWeight < currentWeight) {
    warning = "⚠️ Bạn đang muốn tăng cân nhưng cân nặng mục tiêu lại thấp hơn hiện tại!";
  }

  return (
    <View style={styles.container}>
      {/* BMI hiển thị */}
      <Text style={styles.bmiText}>
        BMI mục tiêu: {bmiValue} ({bmiLevel})
      </Text>

      {/* Slider cân nặng */}
      <Slider
        style={styles.slider}
        minimumValue={minWeight}
        maximumValue={maxWeight}
        step={step}
        value={targetWeight}
        minimumTrackTintColor="#3b82f6"
        maximumTrackTintColor="#d1d5db"
        thumbTintColor="#3b82f6"
        onValueChange={onTargetWeightChange}
      />

      {/* Cân nặng mục tiêu */}
      <Text style={styles.valueText}>{targetWeight} kg</Text>

      {/* Thông báo cảnh báo nếu cần */}
      {warning !== "" && <Text style={styles.warning}>{warning}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { alignItems: "center", marginTop: 30 },
  bmiText: { fontSize: 16, fontWeight: "bold", marginBottom: 10 },
  slider: { width: 300, height: 40 },
  valueText: { fontSize: 16, fontWeight: "bold", marginTop: 10 },
  warning: {
    marginTop: 15,
    fontSize: 14,
    color: "red",
    textAlign: "center",
    paddingHorizontal: 10,
  },
});
