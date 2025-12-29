import React, { useState } from "react";
import { View, Text, TouchableOpacity, StyleSheet, ScrollView, TextInput } from "react-native";

const diseases = [
  "Tiểu đường",
  "Huyết áp cao",
  "Tim mạch",
  "Hen suyễn",
  "Thận",
  "Gan",
  "Ung thư",
  "Khác",
  "Không có bệnh nền",
];

export default function DiseaseTab({ selectedDiseases, onChange }) {
  const [selected, setSelected] = useState(selectedDiseases || []);
  const [otherDisease, setOtherDisease] = useState("");

  const toggleDisease = (disease) => {
    let updated = [...selected];

    if (disease === "Không có bệnh nền") {
      updated = ["Không có bệnh nền"];
      setOtherDisease("");
      onChange(updated, "");
    } else {
      updated = updated.filter((f) => f !== "Không có bệnh nền");

      if (disease === "Khác") {
        if (updated.includes("Khác")) {
          updated = updated.filter((f) => f !== "Khác");
          setOtherDisease("");
          onChange(updated, "");
        } else {
          updated.push("Khác");
          onChange(updated, otherDisease);
        }
      } else {
        if (updated.includes(disease)) {
          updated = updated.filter((f) => f !== disease);
        } else {
          updated.push(disease);
        }
        onChange(updated, otherDisease);
      }
    }

    setSelected(updated);
  };

  const handleOtherChange = (text) => {
    setOtherDisease(text);
    onChange(selected, text);
  };

  return (
    <ScrollView contentContainerStyle={[styles.container, { flexDirection: "row", flexWrap: "wrap" }]}>
      {diseases.map((disease) => {
        const isActive = selected.includes(disease);
        return (
          <View key={disease} style={{ width: "45%", margin: 5 }}>
            <TouchableOpacity
              style={[styles.item, isActive && styles.itemActive]}
              onPress={() => toggleDisease(disease)}
            >
              <Text style={[styles.label, isActive && styles.labelActive]}>{disease}</Text>
            </TouchableOpacity>
            {disease === "Khác" && isActive && (
              <TextInput
                style={styles.input}
                placeholder="Nhập bệnh nền khác..."
                value={otherDisease}
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
    padding: 10,
    justifyContent: "flex-start",
  },
  item: {
    padding: 10,
    borderWidth: 1,
    borderColor: "#ccc",
    borderRadius: 8,
    backgroundColor: "#f9f9f9",
  },
  itemActive: {
    borderColor: "#e53935",
    backgroundColor: "#ffeaea",
  },
  label: {
    textAlign: "center",
  },
  labelActive: {
    color: "#e53935",
    fontWeight: "bold",
  },
  input: {
    marginTop: 5,
    borderWidth: 1,
    borderColor: "#ccc",
    borderRadius: 5,
    padding: 5,
  },
});
