import React from "react";
import { View, Text, StyleSheet } from "react-native";
import Slider from "@react-native-community/slider";

export default function WeightBMISlider({ height, weight, onWeightChange }) {
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

  const bmiValue = calculateBMI(weight);
  const bmiLevel = getBMILevel(bmiValue);

  return (
    <View style={styles.container}>
      {/* BMI hiển thị */}
      <Text style={styles.bmiText}>
        BMI: {bmiValue} ({bmiLevel})
      </Text>

      {/* Slider cân nặng */}
      <Slider
        style={styles.slider}
        minimumValue={minWeight}
        maximumValue={maxWeight}
        step={step}
        value={weight} // ✅ dùng từ props
        minimumTrackTintColor="#3b82f6"
        maximumTrackTintColor="#d1d5db"
        thumbTintColor="#3b82f6"
        onValueChange={onWeightChange} // ✅ gọi callback để update bên ngoài
      />

      {/* Cân nặng hiện tại */}
      <Text style={styles.valueText}>{weight} kg</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { alignItems: "center", marginTop: 50 },
  bmiText: { fontSize: 16, fontWeight: "bold", marginBottom: 10 },
  slider: { width: 300, height: 40 },
  valueText: { fontSize: 16, fontWeight: "bold", marginTop: 10 },
});
