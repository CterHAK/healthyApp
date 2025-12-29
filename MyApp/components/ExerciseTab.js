// components/ExerciseTab.js
import React, { useState } from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";

const activities = [
  { label: "Ít vận động", value: "sedentary", factor: 1.2 },
  { label: "Hoạt động nhẹ", value: "light", factor: 1.375 },
  { label: "Hoạt động vừa phải", value: "moderate", factor: 1.55 },
  { label: "Rất năng động", value: "active", factor: 1.725 },
];

export default function ExerciseTab({ gender, weight, height, age, onResult }) {
  const [selected, setSelected] = useState(null);
  const [bmr, setBmr] = useState(0);
  const [tdee, setTdee] = useState(0);

  const calculate = (activity) => {
    // ✅ Công thức Harris-Benedict
    let bmrCalc =
      gender === "male"
        ? 88.362 + 13.397 * weight + 4.799 * height - 5.677 * age
        : 447.593 + 9.247 * weight + 3.098 * height - 4.330 * age;

    let tdeeCalc = Math.round(bmrCalc * activity.factor);

    setSelected(activity.value);
    setBmr(Math.round(bmrCalc));
    setTdee(tdeeCalc);

    if (onResult) {
      if (onResult) {
        onResult({
          exercise: activity.label,   // hoặc activity.value nếu muốn
          bmr: Math.round(bmrCalc),
          tdee: tdeeCalc,
        });
      }
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Chọn mức độ vận động của bạn</Text>
      {activities.map((act) => {
        const active = selected === act.value;
        return (
          <TouchableOpacity
            key={act.value}
            style={[styles.card, active && styles.cardActive]}
            onPress={() => calculate(act)}
          >
            <Text style={[styles.label, active && styles.labelActive]}>
              {act.label}
            </Text>
          </TouchableOpacity>
        );
      })}

      {bmr > 0 && (
        <View style={styles.resultBox}>
          <Text style={styles.result}>BMR: {bmr} kcal/ngày</Text>
          <Text style={styles.result}>TDEE: {tdee} kcal/ngày</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { width: "100%", padding: 10, alignItems: "center" },
  title: { fontSize: 16, fontWeight: "600", marginBottom: 15 },
  card: {
    width: "90%",
    padding: 15,
    marginVertical: 8,
    borderWidth: 1,
    borderColor: "#ddd",
    borderRadius: 10,
    alignItems: "center",
    backgroundColor: "#fff",
  },
  cardActive: {
    borderColor: "#4CAF50",
    backgroundColor: "#E8F5E9",
  },
  label: { fontSize: 16, color: "#333" },
  labelActive: { fontWeight: "bold", color: "#4CAF50" },
  resultBox: {
    marginTop: 20,
    padding: 15,
    borderRadius: 10,
    backgroundColor: "#f9f9f9",
    borderWidth: 1,
    borderColor: "#ddd",
    width: "90%",
  },
  result: { fontSize: 16, textAlign: "center", marginVertical: 4 },
});
