// screens/ExerciseGroupAddScreen.js
import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  TextInput,
  Button,
  StyleSheet,
  Alert,
  Modal,
  FlatList,
  ScrollView,
  Linking,
} from "react-native";
import { Picker } from "@react-native-picker/picker";
import {
  getMuscles,
  getEquipment,
  filterExercises,
  addExerciseGroupAndDetail,
  getExerciseGroup,
} from "../services/exercise_api";

export default function ExerciseGroupAddScreen({ route, navigation }) {
  const { userData } = route.params || {};
  const email = userData?.email || "";

  const [muscles, setMuscles] = useState([]);
  const [equipment, setEquipment] = useState([]);
  const [filters, setFilters] = useState({ muscle: "", equipment: "", difficulty: "1" });
  const [exercises, setExercises] = useState([]);
  const [selectedExercises, setSelectedExercises] = useState({});
  const [groupModalVisible, setGroupModalVisible] = useState(false);
  const [groupName, setGroupName] = useState("");
  const [groups, setGroups] = useState([]);

  // detail modal
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [exerciseDetail, setExerciseDetail] = useState(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const muscleData = await getMuscles();
        const equipmentData = await getEquipment();
        setMuscles(muscleData.muscle_groups || []);
        setEquipment(equipmentData.equipment_list || []);
        fetchGroups();
      } catch (err) {
        console.error(err);
        Alert.alert("Lỗi", "Không lấy được danh sách nhóm cơ / thiết bị.");
      }
    }
    fetchData();
  }, []);

  const fetchGroups = async () => {
    try {
      const res = await getExerciseGroup({ email });
      if (res?.status === "success") setGroups(res.groups || []);
    } catch (err) {
      console.error(err);
    }
  };

  const handleFilter = async () => {
    try {
      const data = await filterExercises(filters);
      if (data?.status === "success") {
        const initSelection = {};
        (data.results || []).forEach((ex) => {
          initSelection[ex.name] = { selected: false, sets: ex.sets ?? 3, reps: ex.reps ?? 12, expanded: false };
        });
        setSelectedExercises(initSelection);
        setExercises(data.results || []);
      } else {
        setExercises([]);
        Alert.alert("Thông báo", data?.message || "Không tìm thấy bài tập phù hợp.");
      }
    } catch (err) {
      console.error(err);
      setExercises([]);
      Alert.alert("Lỗi", "Không thể lọc bài tập.");
    }
  };

  const toggleSelect = (name) => {
    setSelectedExercises((prev) => ({ ...prev, [name]: { ...prev[name], selected: !prev[name].selected } }));
  };

  const toggleExpand = (name) => {
    setSelectedExercises((prev) => ({ ...prev, [name]: { ...prev[name], expanded: !prev[name].expanded } }));
  };

  const updateSetsReps = (name, field, value) => {
    const normalized = value === "" ? "" : Number(value);
    setSelectedExercises((prev) => ({ ...prev, [name]: { ...prev[name], [field]: normalized } }));
  };

  const handleCreateGroup = async () => {
    if (!groupName.trim()) return Alert.alert("Thông báo", "Vui lòng nhập tên nhóm!");
    const exercisesToAdd = exercises
      .filter((ex) => selectedExercises[ex.name]?.selected)
      .map((ex) => ({ exercise_name: ex.name, sets: selectedExercises[ex.name].sets ?? 3, reps: selectedExercises[ex.name].reps ?? 12 }));
    if (!exercisesToAdd.length) return Alert.alert("Thông báo", "Chưa chọn bài tập nào!");

    try {
      const payload = { email, group_name: groupName.trim(), exercises: exercisesToAdd };
      const res = await addExerciseGroupAndDetail(payload);
      if (res?.status === "success") {
        Alert.alert("✅ Thành công", `${exercisesToAdd.length} bài tập đã thêm vào nhóm "${groupName}"`);
        setGroupModalVisible(false);
        setGroupName("");
        const resetSel = { ...selectedExercises };
        Object.keys(resetSel).forEach((k) => (resetSel[k].selected = false));
        setSelectedExercises(resetSel);
        fetchGroups();
      } else Alert.alert("❌ Lỗi", res?.message || "Thêm bài tập không thành công");
    } catch (err) {
      console.error(err);
      Alert.alert("❌ Lỗi", "Không thể tạo nhóm bài tập.");
    }
  };

  const normalizeInstructions = (raw) => {
    if (!raw) return [];
    if (Array.isArray(raw)) return raw.map(String).filter(Boolean);
    if (typeof raw === "string") return raw.split(/\r?\n|;;|;|\|\||\|/).map(p => p.trim()).filter(Boolean);
    return [];
  };

  const findVideoUrl = (item) => {
    if (!item) return null;
    const keys = ["video", "video_url", "video_link", "videos", "demo", "link"];
    for (const k of keys) {
      const v = item[k];
      if (!v) continue;
      if (typeof v === "string" && v.trim()) return v.trim();
      if (Array.isArray(v) && v.length) return v[0];
    }
    return null;
  };

  const openDetailModal = (item) => {
    const instructions = normalizeInstructions(item.instructions ?? "");
    const video = findVideoUrl(item);
    const combined_muscles = Array.isArray(item.combined_muscles)
      ? item.combined_muscles
      : typeof item.combined_muscles === "string"
      ? item.combined_muscles.split(/,|\|/).map(s => s.trim()).filter(Boolean)
      : [];
    const equipmentArr = Array.isArray(item.equipment)
      ? item.equipment
      : typeof item.equipment === "string"
      ? item.equipment.split(/,|\|/).map(s => s.trim()).filter(Boolean)
      : [];

    setExerciseDetail({ ...item, instructions, video, combined_muscles, equipment: equipmentArr });
    setDetailModalVisible(true);
  };

  const renderExerciseCard = ({ item }) => {
    const sel = selectedExercises[item.name] || {};
    return (
      <View style={styles.card}>
        <TouchableOpacity onPress={() => toggleSelect(item.name)}>
          <Text style={[styles.exerciseTitle, sel.selected && { color: "#28a745" }]}>
            {sel.selected ? "✓ " : ""}{item.name}
          </Text>
        </TouchableOpacity>

        {sel.selected && (
          <>
            <View style={{ flexDirection: "row", gap: 8, marginTop: 8, justifyContent: "space-between" }}>
              <View style={{ alignItems: "center", flex: 1 }}>
                <Text style={styles.label}>Sets</Text>
                <TextInput
                  style={styles.smallInput}
                  keyboardType="numeric"
                  value={sel.sets?.toString() ?? ""}
                  onChangeText={v => updateSetsReps(item.name, "sets", v)}
                  placeholder="Sets"
                />
              </View>
              <View style={{ alignItems: "center", flex: 1 }}>
                <Text style={styles.label}>Reps</Text>
                <TextInput
                  style={styles.smallInput}
                  keyboardType="numeric"
                  value={sel.reps?.toString() ?? ""}
                  onChangeText={v => updateSetsReps(item.name, "reps", v)}
                  placeholder="Reps"
                />
              </View>
            </View>

            <TouchableOpacity style={styles.detailBtn} onPress={() => openDetailModal(item)}>
              <Text style={styles.detailBtnText}>👀 Xem chi tiết</Text>
            </TouchableOpacity>
          </>
        )}
      </View>
    );
  };

  return (
    <View style={{ flex: 1, backgroundColor: "#f8f9fa" }}>
      {/* Header */}
      <View style={{ padding: 16, backgroundColor: "#007bff", borderBottomLeftRadius: 12, borderBottomRightRadius: 12 }}>
        <Text style={{ fontSize: 20, fontWeight: "800", color: "#fff" }}>Bài tập thể thao</Text>

        {/* Tab buttons */}
        <View style={{ flexDirection: "row", marginTop: 12 }}>
          <TouchableOpacity
            style={[styles.tabBtn, { backgroundColor: "#fff" }]}
            onPress={() =>
              navigation.navigate("Home", {
                key: Math.random().toString(), // tạo key mới
                userData,
                selectedTab: "PlanByDate",
              })

            }
          >
            <Text style={{ fontWeight: "700", color: "#007bff" }}>Lịch tập</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.tabBtn, { backgroundColor: "#ffc107", marginLeft: 8 }]}
            onPress={() => setGroupModalVisible(true)}  // ← thêm cái này
          >
            <Text style={{ fontWeight: "700", color: "#fff" }}>Tạo nhóm</Text>
          </TouchableOpacity>

        </View>
      </View>

      {/* Filter */}
      <View style={styles.filterCard}>
        <Text style={styles.filterTitle}>🎯 Lọc bài tập</Text>

        <Text style={styles.filterLabel}>Nhóm cơ:</Text>
        <View style={styles.pickerWrapper}>
          <Picker selectedValue={filters.muscle} onValueChange={v=>setFilters({...filters, muscle:v})} style={styles.picker}>
            <Picker.Item label="-- Chọn --" value="" />
            {muscles.map(m=><Picker.Item key={m} label={m} value={m} />)}
          </Picker>
        </View>

        <Text style={styles.filterLabel}>Thiết bị:</Text>
        <View style={styles.pickerWrapper}>
          <Picker selectedValue={filters.equipment} onValueChange={v=>setFilters({...filters, equipment:v})} style={styles.picker}>
            <Picker.Item label="-- Chọn --" value="" />
            {equipment.map(e=><Picker.Item key={e} label={e} value={e} />)}
          </Picker>
        </View>

        <Text style={styles.filterLabel}>Độ khó:</Text>
        <View style={styles.pickerWrapper}>
          <Picker selectedValue={filters.difficulty} onValueChange={v=>setFilters({...filters, difficulty:v})} style={styles.picker}>
            {[1,2,3,4,5].map(n=><Picker.Item key={n} label={`${"⭐".repeat(n)}`} value={n.toString()} />)}
          </Picker>
        </View>

        <TouchableOpacity style={styles.searchBtn} onPress={handleFilter}>
          <Text style={styles.searchBtnText}>🔍 Tìm bài tập</Text>
        </TouchableOpacity>
      </View>

      {/* List bài tập */}
      <FlatList
        data={exercises}
        keyExtractor={item => item.name}
        renderItem={renderExerciseCard}
        contentContainerStyle={{ padding: 16 }}
      />

      {/* Modal đặt tên nhóm */}
      <Modal transparent animationType="slide" visible={groupModalVisible}>
        <TouchableOpacity style={styles.overlay} onPress={()=>setGroupModalVisible(false)} activeOpacity={1}/>
        <View style={styles.modal}>
          <Text style={{ fontWeight:"700", fontSize:18, marginBottom:10 }}>Tên nhóm bài tập</Text>
          <TextInput placeholder="Nhập tên nhóm..." style={styles.input} value={groupName} onChangeText={setGroupName} />
          <Button title="✅ Tạo nhóm" onPress={handleCreateGroup} />
        </View>
      </Modal>

      {/* Modal flash card */}
      <Modal transparent animationType="slide" visible={detailModalVisible}>
        <View style={styles.modalOverlay}>
          <View style={styles.detailCard}>
            <ScrollView>
              {exerciseDetail ? (
                <>
                  <Text style={styles.detailTitle}>{exerciseDetail.name}</Text>
                  <Text style={styles.sectionTitle}>🎯 Nhóm cơ</Text>
                  <Text style={styles.sectionText}>
                    {exerciseDetail.combined_muscles?.length ? exerciseDetail.combined_muscles.join(", ") : "Chưa có"}
                  </Text>
                  <Text style={styles.sectionTitle}>🏋️ Thiết bị</Text>
                  <Text style={styles.sectionText}>
                    {exerciseDetail.equipment?.length ? exerciseDetail.equipment.join(", ") : "Không cần thiết bị"}
                  </Text>
                  <Text style={styles.sectionTitle}>📘 Hướng dẫn</Text>
                  {exerciseDetail.instructions?.length ? exerciseDetail.instructions.map((ins,i)=><Text key={i} style={styles.stepText}>• {ins}</Text>)
                  : <Text style={styles.sectionText}>📘 Hướng dẫn: Chưa có</Text>}
                  {exerciseDetail.video ? (
                    <TouchableOpacity style={styles.videoBtn} onPress={()=>Linking.openURL(exerciseDetail.video)}>
                      <Text style={styles.videoBtnText}>🎥 Xem video hướng dẫn</Text>
                    </TouchableOpacity>
                  ) : <Text style={{ marginTop:12, color:"#666" }}>🎥 Video: Không có</Text>}
                  <TouchableOpacity style={styles.closeBtn} onPress={()=>setDetailModalVisible(false)}>
                    <Text style={styles.closeBtnText}>Đóng</Text>
                  </TouchableOpacity>
                </>
              ) : <Text>Đang tải...</Text>}
            </ScrollView>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  tabBtn: { flex:1, padding: 8, borderRadius: 8, alignItems: "center", justifyContent: "center" },
  filterCard: { backgroundColor: "#fff", padding: 16, margin: 16, borderRadius: 14, shadowColor: "#000", shadowOpacity: 0.08, shadowRadius: 6, elevation: 4 },
  filterTitle: { fontSize: 20, fontWeight: "800", marginBottom: 12 },
  filterLabel: { fontSize: 14, fontWeight: "600", color: "#444", marginBottom: 6, marginTop: 8 },
  pickerWrapper: { backgroundColor: "#f2f2f2", borderRadius: 10, marginBottom: 8, justifyContent: "center", height: 50 },
  picker: { height: 50, width: "100%" },
  searchBtn: { marginTop: 12, backgroundColor: "#007bff", paddingVertical: 12, borderRadius: 10, shadowColor: "#000", shadowOpacity: 0.1, shadowRadius: 4, elevation: 3 },
  searchBtnText: { color: "#fff", textAlign: "center", fontWeight: "700" },
  card: { padding: 12, marginBottom: 10, backgroundColor: "#fff", borderRadius: 10, borderWidth: 1, borderColor: "#eee", shadowColor: "#000", shadowOpacity: 0.03, elevation: 2 },
  exerciseTitle: { fontSize: 16, fontWeight: "600", color: "#222" },
  smallInput: { borderWidth: 1, borderColor: "#ccc", borderRadius: 6, padding: 6, width: 70, textAlign: "center" },
  label: { fontSize: 13, marginBottom: 4, color: "#555" },
  detailBtn: { marginTop: 10, padding: 10, backgroundColor: "#007bff", borderRadius: 8 },
  detailBtnText: { color: "#fff", textAlign: "center", fontWeight: "600" },
  overlay: { flex: 1, backgroundColor: "rgba(0,0,0,0.3)" },
  modal: { position: "absolute", top: "30%", left: "10%", right: "10%", backgroundColor: "#fff", padding: 20, borderRadius: 14, shadowColor: "#000", shadowOpacity: 0.2, shadowRadius: 10, elevation: 5 },
  input: { borderWidth: 1, borderColor: "#ccc", borderRadius: 10, padding: 10, marginBottom: 12 },
  modalOverlay: { flex: 1, backgroundColor: "rgba(0,0,0,0.5)", justifyContent: "center", alignItems: "center", padding: 16 },
  detailCard: { width: "100%", maxHeight: "85%", backgroundColor: "#fff", borderRadius: 16, padding: 16 },
  detailTitle: { fontSize: 22, fontWeight: "800", marginBottom: 12, textAlign: "center" },
  sectionTitle: { fontSize: 16, fontWeight: "700", marginTop: 10 },
  sectionText: { fontSize: 14, marginTop: 4, color: "#555" },
  stepText: { fontSize: 14, marginTop: 4, marginLeft: 6, color: "#333" },
  videoBtn: { marginTop: 12, padding: 10, backgroundColor: "#28a745", borderRadius: 8 },
  videoBtnText: { color: "#fff", fontWeight: "600", textAlign: "center" },
  closeBtn: { marginTop: 16, padding: 10, backgroundColor: "#dc3545", borderRadius: 8 },
  closeBtnText: { color: "#fff", fontWeight: "700", textAlign: "center" },
});
