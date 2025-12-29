import React from "react";
import { View, TouchableOpacity, Text, StyleSheet } from "react-native";

export default function RadioGroup({ options, value, onChange }) {
  return (
    <View style={styles.radioGroup}>
      {options.map((option) => (
        <TouchableOpacity
          key={option}
          style={styles.radioOption}
          onPress={() => onChange(option)}
        >
          <View
            style={[
              styles.radioCircle,
              value === option && styles.radioCircleSelected,
            ]}
          />
          <Text>{option}</Text>
        </TouchableOpacity>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  radioGroup: { flexDirection: "row", marginBottom: 20 },
  radioOption: { flexDirection: "row", alignItems: "center", marginHorizontal: 10 },
  radioCircle: { width: 18, height: 18, borderRadius: 9, borderWidth: 2, borderColor: "#4caf50", marginRight: 6 },
  radioCircleSelected: { backgroundColor: "#4caf50" },
});