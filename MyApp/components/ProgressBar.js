import React from "react";
import { View, StyleSheet } from "react-native";

export default function ProgressBar({ steps, step }) {
  return (
    <View style={styles.progressBar}>
      {steps.map((_, i) => (
        <View
          key={i}
          style={[
            styles.progressStep,
            i <= step && styles.progressStepActive,
          ]}
        />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  progressBar: { flexDirection: "row", marginBottom: 30 },
  progressStep: { width: 30, height: 8, backgroundColor: "#ccc", margin: 4, borderRadius: 4 },
  progressStepActive: { backgroundColor: "#4caf50" },
});