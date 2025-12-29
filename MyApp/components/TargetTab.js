// components/TargetTab.js
import React from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";

const targets = [
  { label: "Giảm cân", value: "lose_weight" },
  { label: "Tăng cân", value: "gain_weight" },
  { label: "Cải thiện sức khoẻ", value: "health" },
];

export default function TargetTab({ selectedTarget, onSelect }) {
  return (
    <View style={styles.container}>
      {targets.map((target) => {
        const isActive = selectedTarget === target.value;
        return (
          <TouchableOpacity
            key={target.value}
            style={[styles.card, isActive && styles.cardActive]}
            onPress={() => onSelect(target.value)}
          >
            <Text style={[styles.label, isActive && styles.labelActive]}>
              {target.label}
            </Text>
          </TouchableOpacity>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    justifyContent: "space-between",
    width: "100%",
    marginVertical: 20,
    paddingHorizontal: 10,
  },
  card: {
    flex: 1,
    marginHorizontal: 5,
    paddingVertical: 20,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#fff",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#ddd",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 4,
    elevation: 3, // cho Android
  },
  cardActive: {
    borderColor: "#4CAF50",
    backgroundColor: "#E8F5E9",
    shadowOpacity: 0.3,
    elevation: 5,
  },
  label: {
    fontSize: 16,
    color: "#111",
  },
  labelActive: {
    fontWeight: "bold",
    color: "#4CAF50",
  },
});
