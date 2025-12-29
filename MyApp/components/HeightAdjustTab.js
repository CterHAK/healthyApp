import React, { useState } from "react";
import { View, Text, StyleSheet } from "react-native";
import Slider from "@react-native-community/slider";

export default function TickSlider() {
  const [value, setValue] = useState(120);

  const min = 120;
  const max = 300;
  const step = 10;
  const ticks = [];
  for (let i = max; i >= min; i -= step) {
    ticks.push(i);
  }

  return (
    <View style={styles.container}>
      {/* Slider and ticks wrapper */}
      <View style={styles.sliderAndTicks}>
        {/* Slider wrapped in a View with fixed dimensions */}
        <View style={styles.sliderContainer}>
          <Slider
            style={styles.slider}
            minimumValue={min}
            maximumValue={max}
            step={step}
            value={value}
            minimumTrackTintColor="#3b82f6"
            maximumTrackTintColor="#d1d5db"
            thumbTintColor="#3b82f6"
            onValueChange={setValue}
          />
        </View>

        {/* Tick marks and labels */}
        <View style={styles.ticksContainer}>
          {ticks.map((tick, index) => (
            <View key={index} style={styles.tickWrapper}>
              <Text style={styles.tickLabel}>{tick}</Text>
              <View style={styles.tick} />
            </View>
          ))}
        </View>
      </View>

      {/* Current value display */}
      <Text style={styles.valueText}>Current Value: {value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: "column", // Stack slider/ticks and value text vertically
    marginLeft: 50,
    marginTop: 20,
    alignItems: "center",
  },
  sliderAndTicks: {
    flexDirection: "row", // Slider and ticks side by side
    height: 300, // Height for vertical slider
    alignItems: "center",
  },
  sliderContainer: {
    width: 50, // Fixed width for the slider container
    height: 320, // Fixed height for the slider container
    justifyContent: "center",
    alignItems: "center",
  },
  slider: {
    height: 300, // Vertical height
    width: 320, // Thickness of the slider
    transform: [{ rotate: "-90deg" }], // Rotate to make vertical
  },
  ticksContainer: {
    justifyContent: "space-between",
    height: 300, // Match slider height
    marginLeft: 10, // Space between slider and ticks
  },
  tickWrapper: {
    flexDirection: "row", // Label and tick side by side
    alignItems: "center",
  },
  tick: {
    width: 10, // Length of tick mark
    height: 2, // Thickness of tick mark
    backgroundColor: "black",
    marginLeft: 4, // Space between label and tick
  },
  tickLabel: {
    fontSize: 10,
    color: "#111",
  },
  valueText: {
    marginTop: 10, // Space between slider and text
    fontSize: 14,
    color: "#111",
    fontWeight: "bold",
  },
});