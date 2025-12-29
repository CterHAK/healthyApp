// screens/PlanByDateScreen.js
import React, { useState, useEffect, useCallback } from "react";
import { 
  View, Text, ScrollView, StyleSheet, TouchableOpacity, Alert, Modal, Linking 
} from "react-native";
import DateTimePicker from "@react-native-community/datetimepicker";
import { useFocusEffect } from "@react-navigation/native";
import { 
  getExerciseGroup, addExercisePlan, getExercisePlans, getExerciseInfo, markWorkoutDone 
} from "../services/exercise_api";

export default function PlanByDateScreen({ route, navigation }) {
  const { userData } = route.params;
  const email = userData.email;

  const [groups, setGroups] = useState([]);
  const [expandedGroup, setExpandedGroup] = useState(null);
  const [selectedDate, setSelectedDate] = useState(new Date());
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [plans, setPlans] = useState([]);
  const [selectedSession, setSelectedSession] = useState("Sáng");
  const sessions = ["Sáng","Trưa","Chiều","Tối"];

  const [modalVisible, setModalVisible] = useState(false);
  const [exerciseDetail, setExerciseDetail] = useState(null);

  // === Lấy nhóm bài tập ===
  const fetchGroups = async () => {
    try {
      const res = await getExerciseGroup({ email });
      if (res.status === "success") setGroups(res.groups);
    } catch (e) { console.error(e); }
  };

  // === Lấy kế hoạch tập theo ngày ===
  const fetchPlans = async (date) => {
    try {
      const dayStr = date.toISOString().split("T")[0];
      const res = await getExercisePlans({ email, day: dayStr });
      if (res.status === "success") setPlans(res.plans);
    } catch (e) { console.error(e); }
  };

  const onChangeDate = (event, date) => {
    setShowDatePicker(false);
    if (date) setSelectedDate(date);
  };

  const handleAddPlan = async (group_name) => {
    try {
      const dayStr = selectedDate.toISOString().split("T")[0];
      const res = await addExercisePlan({
        email, group_name, day: dayStr, session: selectedSession, done_flag:false
      });
      if (res.status === "success") {
        Alert.alert("Thành công", "Đã thêm vào plan");
        fetchPlans(selectedDate);
      } else Alert.alert("Lỗi", res.message);
    } catch (e) { console.error(e); }
  };

  const showExerciseInfo = async (exercise_name) => {
    try {
      const res = await getExerciseInfo(exercise_name);
      if (res.status === "success") {
        setExerciseDetail(res.exercise);
        setModalVisible(true);
      } else Alert.alert("Lỗi", res.message);
    } catch (e) { console.error(e); }
  };

  useFocusEffect(
    useCallback(() => {
      fetchGroups();
      fetchPlans(selectedDate);
    }, [selectedDate])
  );

  return (
    <View style={{flex:1, backgroundColor:"#f8f9fa"}}>
      {/* Header Tab */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>🏋️‍♂️ Lịch tập thể thao</Text>
        <View style={styles.tabs}>
          <TouchableOpacity style={[styles.tabBtn, {backgroundColor:"#ffffffff"}]} onPress={()=>navigation.navigate("ExerciseGroupAdd", { userData })}>
            <Text style={{color:"#007bff", fontWeight:"700"}}>Lọc bài tập</Text>
          </TouchableOpacity>

        </View>
      </View>

      {/* Chọn ngày */}
      <View style={{padding:16}}>
        <Text style={styles.sectionTitle}>📅 Chọn ngày</Text>
        <TouchableOpacity style={styles.dateBtn} onPress={()=>setShowDatePicker(true)}>
          <Text style={styles.dateBtnText}>{selectedDate.toDateString()}</Text>
        </TouchableOpacity>
        {showDatePicker && <DateTimePicker value={selectedDate} mode="date" display="calendar" onChange={onChangeDate} />}
      </View>

      {/* Nhóm bài tập */}
      <Text style={[styles.sectionTitle,{paddingLeft:16}]}>🏋️ Nhóm bài tập</Text>
      <ScrollView contentContainerStyle={{paddingHorizontal:16, paddingBottom:20}}>
        {groups.map(item => (
          <View key={item.group.group_name} style={styles.card}>
            <TouchableOpacity onPress={()=>setExpandedGroup(expandedGroup===item.group.group_name ? null : item.group.group_name)}>
              <Text style={styles.cardTitle}>{item.group.group_name}</Text>
            </TouchableOpacity>

            {expandedGroup===item.group.group_name && (
              <View style={{marginTop:10}}>
                {item.details.map(ex => (
                  <TouchableOpacity key={ex._id} onPress={()=>showExerciseInfo(ex.name)}>
                    <Text style={styles.exerciseText}>• {ex.name} ({ex.sets}x{ex.reps})</Text>
                  </TouchableOpacity>
                ))}

                <Text style={{marginTop:10,fontWeight:"600"}}>Buổi tập:</Text>
                <View style={{flexDirection:'row', marginVertical:5}}>
                  {sessions.map(s => (
                    <TouchableOpacity 
                      key={s}
                      style={[styles.sessionBtn, s===selectedSession && styles.sessionBtnSelected]}
                      onPress={()=>setSelectedSession(s)}
                    >
                      <Text style={{color:s===selectedSession?"#fff":"#000"}}>{s}</Text>
                    </TouchableOpacity>
                  ))}
                </View>

                <TouchableOpacity style={styles.addBtn} onPress={()=>handleAddPlan(item.group.group_name)}>
                  <Text style={styles.addBtnText}>➕ Thêm vào Plan</Text>
                </TouchableOpacity>
              </View>
            )}
          </View>
        ))}
      </ScrollView>

      {/* Lịch tập hôm nay */}
      <Text style={[styles.sectionTitle,{paddingLeft:16}]}>📋 Lịch tập hôm {selectedDate.toDateString()}</Text>
      <ScrollView contentContainerStyle={{paddingHorizontal:16, paddingBottom:50}}>
        {plans.map(plan => (
          <View key={plan._id} style={styles.card}>
            <Text style={styles.cardTitle}>{plan.group_name}</Text>
            <Text>Buổi: {plan.session}</Text>
            {plan.details?.map(d => <Text key={d._id} style={styles.exerciseText}>• {d.name} ({d.sets}x{d.reps})</Text>)}

            {plan.done_flag ? (
              <View style={[styles.doneBtn]}><Text style={{color:"#fff"}}>Đã hoàn thành</Text></View>
            ) : (
              <TouchableOpacity style={styles.doneBtnPending} onPress={async ()=>{
                try {
                  const dayStr = selectedDate.toISOString().split("T")[0];
                  const res = await markWorkoutDone({
                    email,
                    group_name: plan.group_name,
                    day: dayStr,
                    session: plan.session,
                    weight: userData.weight || 70
                  });
                  if (res.success) fetchPlans(selectedDate);
                  else Alert.alert("Lỗi", res.message || "Không thể cập nhật");
                } catch(e){console.error(e);}
              }}>
                <Text style={{color:"#fff"}}>Đánh dấu đã tập</Text>
              </TouchableOpacity>
            )}
          </View>
        ))}
      </ScrollView>

      {/* Modal chi tiết bài tập */}
      <Modal visible={modalVisible} animationType="slide" transparent>
        <View style={styles.modalOverlay}>
          <View style={styles.detailCard}>
            <ScrollView>
              {exerciseDetail ? (
                <>
                  <Text style={styles.detailTitle}>{exerciseDetail.name}</Text>
                  <Text style={styles.sectionSubTitle}>Mô tả</Text>
                  <Text>{exerciseDetail.instructions || "Không có mô tả"}</Text>
                  <Text style={styles.sectionSubTitle}>Nhóm cơ</Text>
                  <Text>{Array.isArray(exerciseDetail.muscles) ? exerciseDetail.muscles.join(", ") : exerciseDetail.muscles}</Text>
                  <Text style={styles.sectionSubTitle}>Thiết bị</Text>
                  <Text>{exerciseDetail.equipment}</Text>
                  <Text style={styles.sectionSubTitle}>MET</Text>
                  <Text>{exerciseDetail.MET}</Text>

                  {exerciseDetail.video && (
                    <TouchableOpacity style={styles.videoBtn} onPress={()=>Linking.openURL(exerciseDetail.video)}>
                      <Text style={{color:"#fff", textAlign:"center"}}>🎥 Xem video</Text>
                    </TouchableOpacity>
                  )}
                  <TouchableOpacity style={styles.closeBtn} onPress={()=>setModalVisible(false)}>
                    <Text style={{textAlign:"center"}}>Đóng</Text>
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
  header:{padding:16, backgroundColor:"#007bff", borderBottomLeftRadius:12, borderBottomRightRadius:12},
  headerTitle:{fontSize:20,fontWeight:"800",color:"#fff"},
  tabs:{flexDirection:"row",marginTop:12},
  tabBtn:{flex:1,padding:8,borderRadius:8,alignItems:"center",justifyContent:"center"},

  sectionTitle:{fontSize:18,fontWeight:"bold",marginVertical:10},
  dateBtn:{backgroundColor:"#007bff", padding:10, borderRadius:10, marginBottom:10},
  dateBtnText:{color:"#fff", textAlign:"center", fontWeight:"600"},

  card:{backgroundColor:"#fff", borderRadius:12, padding:12, marginBottom:12, shadowColor:"#000", shadowOpacity:0.05, shadowRadius:6, elevation:3},
  cardTitle:{fontSize:16, fontWeight:"700"},
  exerciseText:{marginLeft:8, marginVertical:2, color:"#444"},

  sessionBtn:{paddingVertical:6, paddingHorizontal:12, borderWidth:1, borderColor:"#ccc", borderRadius:20, marginRight:5},
  sessionBtnSelected:{backgroundColor:"#007bff", borderColor:"#007bff"},

  addBtn:{backgroundColor:"#28a745", padding:10, borderRadius:8, marginTop:10},
  addBtnText:{color:"#fff", textAlign:"center", fontWeight:"600"},

  doneBtn:{backgroundColor:"#28a745", padding:8, borderRadius:6, marginTop:6},
  doneBtnPending:{backgroundColor:"#007bff", padding:8, borderRadius:6, marginTop:6},

  modalOverlay:{flex:1, backgroundColor:"rgba(0,0,0,0.5)", justifyContent:"center", padding:16},
  detailCard:{backgroundColor:"#fff", borderRadius:12, padding:16, maxHeight:"85%"},
  detailTitle:{fontSize:20,fontWeight:"700",marginBottom:8},
  sectionSubTitle:{fontWeight:"700", marginTop:10, marginBottom:4},
  videoBtn:{backgroundColor:"#28a745", padding:10, borderRadius:8, marginTop:10},
  closeBtn:{backgroundColor:"#ccc", padding:10, borderRadius:8, marginTop:12}
});
