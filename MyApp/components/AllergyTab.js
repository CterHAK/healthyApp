import React, { useState } from "react";
import { View, Text, TouchableOpacity, StyleSheet, ScrollView, TextInput } from "react-native";

const foods = [
  "Sữa",
  "Trứng",
  "Đậu phộng",
  "Hải sản",
  "Đậu nành",
  "Lúa mì",
  "Cây hạt (óc chó, hạnh nhân...)",
  "Dâu tây",
  "Mật ong",
  "Khác",          // ✅ thêm
  "Không dị ứng",   // ✅ thêm
];

export default function AllergyTab({ selectedFoods, onChange }) {
  const [selected, setSelected] = useState(selectedFoods || []);
  const [otherFood, setOtherFood] = useState("");

  const toggleFood = (food) => {
    let updated = [...selected];

    if (food === "Không dị ứng") {
      // ✅ chọn "Không dị ứng" thì xoá hết, chỉ giữ lại cái này
      updated = ["Không dị ứng"];
      setOtherFood(""); // clear input khác
    } else {
      // Nếu đang chọn "Không dị ứng" thì bỏ nó đi khi chọn cái khác
      updated = updated.filter((f) => f !== "Không dị ứng");

      if (food === "Khác") {
        // toggle Khác
        if (updated.includes("Khác")) {
          updated = updated.filter((f) => f !== "Khác");
          setOtherFood(""); // clear khi bỏ chọn
        } else {
          updated.push("Khác");
        }
      } else {
        // toggle normal
        if (updated.includes(food)) {
          updated = updated.filter((f) => f !== food);
        } else {
          updated.push(food);
        }
      }
    }

    setSelected(updated);
    onChange(updated, otherFood);
  };

  const handleOtherChange = (text) => {
    setOtherFood(text);
    onChange(selected, text);
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      {foods.map((food) => {
        const isActive = selected.includes(food);
        return (
          <View key={food} style={{ width: "45%", margin: 5 }}>
            <TouchableOpacity
              style={[styles.item, isActive && styles.itemActive]}
              onPress={() => toggleFood(food)}
            >
              <Text style={[styles.label, isActive && styles.labelActive]}>{food}</Text>
            </TouchableOpacity>
            {food === "Khác" && isActive && (
              <TextInput
                style={styles.input}
                placeholder="Nhập thực phẩm khác..."
                value={otherFood}
                onChangeText={handleOtherChange}
              />
            )}
          </View>
        );
      })}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "center",
    padding: 10,
  },
  item: {
    paddingVertical: 10,
    paddingHorizontal: 15,
    borderWidth: 1,
    borderColor: "#ccc",
    borderRadius: 20,
    backgroundColor: "#fff",
    alignItems: "center",
  },
  itemActive: {
    backgroundColor: "#e3f2fd",
    borderColor: "#2196f3",
  },
  label: {
    fontSize: 14,
    color: "#333",
  },
  labelActive: {
    fontWeight: "bold",
    color: "#1976d2",
  },
  input: {
    marginTop: 5,
    borderWidth: 1,
    borderColor: "#ccc",
    borderRadius: 10,
    paddingHorizontal: 10,
    height: 40,
    fontSize: 14,
  },
});
