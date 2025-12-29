// components/ExerciseQuickAddSheet.js
import React, { useState } from "react";
import {
  View,
  Text,
  Modal,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  Alert
} from "react-native";
import { Picker } from "@react-native-picker/picker";
import { addExercisePlan } from "../services/exercise_api";

export default function ExerciseQuickAddSheet({
  visible,
  onClose,
  exercise,
  selectedDate
}) {
  const [sets, setSets] = useState(exercise?.sets?.toString() || "3");
  const [reps, setReps] = useState(exercise?.reps?.toString() || "10");
  const [session, setSession] = useState("Sáng");

  if (!exercise) return null;

  const handleAdd = async () => {
    try {
      await addExercisePlan({
        name: exercise.name,
        muscle: exercise.combined_muscles?.join(", ") || "N/A",
        category: exercise.category,
        equipment: exercise.equipment?.join(", ") || "N/A",
        video: exercise.video,
        instructions: exercise.instructions,
        cluster_label: exercise.cluster_label,
        MET: exercise.MET,
        sets: Number(sets),
        reps: Number(reps),
        day: selectedDate,
        session,
      });
      Alert.alert("✅ Thêm thành công", `${exercise.name} đã được thêm vào lịch!`);
      onClose();
    } catch (err) {
      Alert.alert("❌ Lỗi", "Không thể thêm bài tập. Thử lại!");
      console.error(err);
    }
  };

  return (
    <Modal transparent animationType="slide" visible={visible}>
      <TouchableOpacity style={styles.overlay} onPress={onClose} activeOpacity={1} />
      <View style={styles.sheet}>
        <Text style={styles.title}>{exercise.name}</Text>
        <Text style={styles.line}>💪 {exercise.combined_muscles?.join(", ")}</Text>
        <Text style={styles.line}>🏋️ {exercise.equipment?.join(", ")}</Text>
        <Text style={styles.line}>⭐ Độ khó: {exercise.difficulty}</Text>

        <Text style={styles.label}>Sets</Text>
        <TextInput
          style={styles.input}
          value={sets}
          onChangeText={setSets}
          keyboardType="numeric"
        />

        <Text style={styles.label}>Reps</Text>
        <TextInput
          style={styles.input}
          value={reps}
          onChangeText={setReps}
          keyboardType="numeric"
        />

        <Text style={styles.label}>Buổi tập</Text>
        <Picker
          selectedValue={session}
          onValueChange={(v) => setSession(v)}
          style={styles.picker}
        >
          <Picker.Item label="Sáng" value="Sáng" />
          <Picker.Item label="Trưa" value="Trưa" />
          <Picker.Item label="Chiều" value="Chiều" />
        </Picker>

        <TouchableOpacity style={styles.addBtn} onPress={handleAdd}>
          <Text style={styles.addText}>➕ Thêm vào lịch</Text>
        </TouchableOpacity>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: { flex: 1, backgroundColor: "rgba(0,0,0,0.3)" },
  sheet: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    padding: 16,
    backgroundColor: "#fff",
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
  },
  title: { fontSize: 20, fontWeight: "700", marginBottom: 8 },
  line: { opacity: 0.7, marginBottom: 4 },
  label: { marginTop: 10, fontWeight: "600" },
  input: {
    borderWidth: 1,
    borderColor: "#ccc",
    borderRadius: 6,
    padding: 8,
    marginTop: 4,
  },
  picker: { marginTop: -8 },
  addBtn: {
    marginTop: 20,
    backgroundColor: "#28a745",
    padding: 12,
    borderRadius: 8,
    alignItems: "center",
  },
  addText: { color: "#fff", fontWeight: "700" },
});