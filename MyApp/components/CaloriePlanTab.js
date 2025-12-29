import React, { useEffect } from "react";
import { View, Text, StyleSheet } from "react-native";

export default function CaloriePlanTab({ weight, targetWeight, tdee, onChange }) {
  if (!tdee) {
    return (
      <View style={styles.container}>
        <Text style={styles.warning}>
          ⚠️ Hãy nhập thông tin vận động để tính BMR & TDEE trước.
        </Text>
      </View>
    );
  }

  const diffKg = targetWeight - weight;
  const diffCalories = diffKg * 7700;
  const weeks = 8;
  const weeklyCalories = diffCalories / weeks;
  const dailyCalories = weeklyCalories / 7;

  const recommendedCalories =
    diffKg < 0
      ? tdee + dailyCalories
      : tdee + dailyCalories;

  // ✅ Chỉ gọi onChange khi weight/targetWeight/tdee thay đổi
  useEffect(() => {
    if (onChange) {
      onChange(Math.round(recommendedCalories));
    }
  }, [weight, targetWeight, tdee]);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>📊 Kế hoạch Calories</Text>
      <Text style={styles.text}>Cân nặng hiện tại: {weight} kg</Text>
      <Text style={styles.text}>Cân nặng mục tiêu: {targetWeight} kg</Text>
      <Text style={styles.text}>TDEE hiện tại: {tdee.toFixed(0)} kcal/ngày</Text>
      <Text style={styles.text}>
        Tổng năng lượng cần thay đổi: {Math.abs(diffCalories).toFixed(0)} kcal
      </Text>
      <Text style={styles.text}>
        Kế hoạch ({weeks} tuần): {Math.abs(weeklyCalories).toFixed(0)} kcal/tuần
      </Text>
      <Text style={styles.text}>
        Lượng khuyến nghị hàng ngày: {recommendedCalories.toFixed(0)} kcal/ngày
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { padding: 20, alignItems: "center" },
  title: { fontSize: 20, fontWeight: "bold", marginBottom: 10 },
  text: { fontSize: 16, marginVertical: 4 },
  warning: { fontSize: 16, color: "red", textAlign: "center" },
});
