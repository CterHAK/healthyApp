import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  ActivityIndicator,
  TouchableOpacity,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { fetchTodayCalories } from "../services/exercise_api"; // Đảm bảo đường dẫn đúng

export default function DashboardScreen({ route, navigation }) {
  const { userData } = route.params || {};
  const [todayCalories, setTodayCalories] = useState(null);
  const [loadingCalories, setLoadingCalories] = useState(false);

  const fetchesTodayCaloriesData = async () => {
    setLoadingCalories(true);
    const data = await fetchTodayCalories(userData.email, userData.weight || 70);
    if (data) setTodayCalories(data);
    setLoadingCalories(false);
  };

  if (!userData) {
    return (
      <View style={styles.container}>
        <Text style={styles.header}>Không có dữ liệu người dùng</Text>
      </View>
    );
  }

  useEffect(() => {
    if (userData?.email) fetchesTodayCaloriesData();
  }, [userData]);



  // Chuẩn hóa dữ liệu user
  const user = {
    ...userData,
    exercise: userData.exercise || "—",
    caloriePlan: userData.caloriePlan || "—",
    allergies: Array.isArray(userData.allergies) ? userData.allergies : [],
    diseases: Array.isArray(userData.diseases) ? userData.diseases : [],
  };

  const bmi = (user.weight / ((user.height / 100) ** 2)).toFixed(1);
  let bmiStatus = "";
  if (bmi < 18.5) bmiStatus = "Thiếu cân";
  else if (bmi < 24.9) bmiStatus = "Bình thường";
  else if (bmi < 29.9) bmiStatus = "Thừa cân";
  else bmiStatus = "Béo phì";

  // Tính tổng tiêu hao thực tế (BMR + tập hôm nay)
  const actualExpenditure =
    (user.bmr || 0) + (todayCalories?.burned?.total_calories || 0);

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.header}>Tổng quan sức khỏe</Text>

      {/* Thông tin cá nhân */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Thông tin cá nhân</Text>
        <Text>Họ tên: {user.name}</Text>
        <Text>Email: {user.email}</Text>
        <Text>Giới tính: {user.gender}</Text>
        <Text>Tuổi: {user.age}</Text>
      </View>

      {/* Thông tin thể chất */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Thông tin thể chất</Text>
        <Text>Chiều cao: {user.height} cm</Text>
        <Text>Cân nặng hiện tại: {user.weight} kg</Text>
        <Text>Cân nặng mục tiêu: {user.targetWeight} kg</Text>
        <Text>Mục tiêu: {user.target || "—"}</Text>
        <Text>BMI: {bmi} ({bmiStatus})</Text>
      </View>

      {/* Dị ứng & bệnh nền */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Dị ứng & Bệnh nền</Text>
        <Text>Dị ứng: {user.allergies.length > 0 ? user.allergies.join(", ") : "Không"}</Text>
        <Text>Bệnh nền: {user.diseases.length > 0 ? user.diseases.join(", ") : "Không"}</Text>
      </View>

      {/* Card Calo hôm nay – PHẦN QUAN TRỌNG NHẤT */}
      <View style={styles.calorieCard}>
        <View style={styles.calorieHeader}>
          <Text style={styles.cardTitle}>Calo & Năng Lượng Hôm Nay</Text>
          <TouchableOpacity onPress={fetchesTodayCaloriesData} disabled={loadingCalories}>
            <Ionicons name="refresh" size={20} color="#4ECDC4" />
          </TouchableOpacity>
        </View>

        {/* BMR / TDEE */}
        {user.bmr > 0 && (
          <View style={styles.energyInfoContainer}>
            <View style={styles.energyInfoRow}>
              <Text style={styles.energyLabel}>BMR (Chuyển hóa cơ bản):</Text>
              <Text style={styles.energyValue}>{Math.round(user.bmr)} kcal/ngày</Text>
            </View>
            <View style={styles.energyInfoRow}>
              <Text style={styles.energyLabel}>TDEE (Ước tính):</Text>
              <Text style={styles.energyValue}>{Math.round(user.tdee)} kcal/ngày</Text>
            </View>
            <View style={styles.energyInfoRow}>
              <Text style={styles.energyLabel}>Mức độ vận động:</Text>
              <Text style={styles.energyValue}>{user.exercise}</Text>
            </View>
          </View>
        )}

        {loadingCalories ? (
          <ActivityIndicator size="small" color="#4ECDC4" style={{ marginVertical: 20 }} />
        ) : todayCalories ? (
          <>
            {/* 3 cột chính */}
            <View style={styles.calorieRow}>
              <View style={styles.calorieItem}>
                <Text style={styles.calorieLabel}>Ăn uống</Text>
                <Text style={styles.calorieValue}>
                  {Math.round(todayCalories.intake?.total_calories || 0)}
                </Text>
                <Text style={styles.calorieUnit}>kcal</Text>
              </View>
              <View style={styles.divider} />
              <View style={styles.calorieItem}>
                <Text style={styles.calorieLabel}>Tập luyện</Text>
                <Text style={styles.calorieValue}>
                  {Math.round(todayCalories.burned?.total_calories || 0)}
                </Text>
                <Text style={styles.calorieUnit}>kcal</Text>
              </View>
              <View style={styles.divider} />
              <View style={styles.calorieItem}>
                <Text style={styles.calorieLabel}>Net calories</Text>
                <Text
                  style={[
                    styles.calorieValue,
                    { color: todayCalories.net_calories < 0 ? "#FF6B6B" : "#4ECDC4" },
                  ]}
                >
                  {Math.round(todayCalories.net_calories || 0)}
                </Text>
                <Text style={styles.calorieUnit}>kcal</Text>
              </View>
            </View>

            {/* Chi tiết tiêu hao + Progress bar */}
            {actualExpenditure > 0 && (
              <View style={styles.comparisonContainer}>
                <Text style={styles.breakdownTitle}>Chi tiết tiêu hao năng lượng:</Text>

                <View style={styles.comparisonRow}>
                  <Text style={styles.comparisonLabel}>BMR (tĩnh):</Text>
                  <Text style={styles.comparisonValue}>{Math.round(user.bmr || 0)} kcal</Text>
                </View>
                <View style={styles.comparisonRow}>
                  <Text style={styles.comparisonLabel}>Từ tập luyện hôm nay:</Text>
                  <Text style={styles.comparisonValue}>
                    {Math.round(todayCalories.burned?.total_calories || 0)} kcal
                  </Text>
                </View>
                <View style={styles.comparisonRow}>
                  <Text style={[styles.comparisonLabel, { fontWeight: "700", color: "#e74c3c" }]}>
                    Tổng tiêu hao (Actual):
                  </Text>
                  <Text style={[styles.comparisonValue, { fontWeight: "700", color: "#e74c3c" }]}>
                    {Math.round(actualExpenditure)} kcal
                  </Text>
                </View>

                {/* Progress bar */}
                <View style={styles.progressBarContainer}>
                  <View
                    style={[
                      styles.progressBar,
                      {
                        width: `${Math.min(
                          ((todayCalories.intake?.total_calories || 0) / actualExpenditure) * 100,
                          100
                        )}%`,
                        backgroundColor:
                          (todayCalories.intake?.total_calories || 0) > actualExpenditure
                            ? "#FF6B6B"
                            : "#4ECDC4",
                      },
                    ]}
                  />
                </View>
                <View style={styles.comparisonRow}>
                  <Text style={styles.comparisonLabel}>Nhận vào / Tiêu hao:</Text>
                  <Text
                    style={[
                      styles.comparisonValue,
                      {
                        color:
                          (todayCalories.intake?.total_calories || 0) > actualExpenditure
                            ? "#FF6B6B"
                            : "#27ae60",
                      },
                    ]}
                  >
                    {Math.round(todayCalories.intake?.total_calories || 0)} /{" "}
                    {Math.round(actualExpenditure)} kcal
                  </Text>
                </View>
              </View>
            )}

            {/* Lịch sử ăn + tập */}
            {(todayCalories.intake?.meals?.length > 0 ||
              todayCalories.burned?.exercises?.length > 0) && (
              <View style={styles.breakdownContainer}>
                {todayCalories.intake?.meals?.length > 0 && (
                  <View style={styles.breakdownSection}>
                    <Text style={styles.breakdownTitle}>Lịch sử ăn uống:</Text>
                    {todayCalories.intake.meals.map((meal, i) => (
                      <Text key={i} style={styles.breakdownItem}>
                        • {meal.session}: {meal.name} ({Math.round(meal.calories)} kcal)
                      </Text>
                    ))}
                  </View>
                )}

                {todayCalories.burned?.exercises?.length > 0 && (
                  <View style={styles.breakdownSection}>
                    <Text style={styles.breakdownTitle}>Lịch sử tập luyện:</Text>
                    {todayCalories.burned.exercises.map((ex, i) => (
                      <Text key={i} style={styles.breakdownItem}>
                        • {ex.session}: {ex.group_name} ({Math.round(ex.calories)} kcal)
                      </Text>
                    ))}
                  </View>
                )}
              </View>
            )}
          </>
        ) : (
          <Text style={styles.emptyText}>Chưa có dữ liệu hôm nay</Text>
        )}
      </View>

      {/* Nút đăng xuất */}
      <TouchableOpacity
        style={styles.logoutButton}
        onPress={() => navigation.replace("Login")}
      >
        <Ionicons name="log-out-outline" size={20} color="#fff" />
        <Text style={styles.logoutText}>Đăng xuất</Text>
      </TouchableOpacity>

      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

// Styles giữ nguyên 100%
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f9f9f9", padding: 16 },
  header: { fontSize: 22, fontWeight: "bold", textAlign: "center", marginBottom: 16, color: "#1e88e5" },
  card: { backgroundColor: "#fff", padding: 16, borderRadius: 12, marginBottom: 14, shadowColor: "#000", shadowOpacity: 0.1, shadowRadius: 5, elevation: 3 },
  cardTitle: { fontSize: 18, fontWeight: "600", marginBottom: 8, color: "#333" },

  calorieCard: { backgroundColor: "#fff", borderRadius: 12, padding: 16, marginBottom: 14, shadowColor: "#000", shadowOpacity: 0.1, shadowRadius: 5, elevation: 3 },
  calorieHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 12 },
  calorieRow: { flexDirection: "row", justifyContent: "space-around", alignItems: "center", marginBottom: 12 },
  calorieItem: { alignItems: "center", flex: 1 },
  calorieLabel: { fontSize: 12, color: "#666", marginBottom: 4 },
  calorieValue: { fontSize: 24, fontWeight: "700", color: "#2a9d8f" },
  calorieUnit: { fontSize: 11, color: "#999", marginTop: 2 },
  divider: { width: 1, height: 50, backgroundColor: "#EEE" },

  breakdownContainer: { marginTop: 12, paddingTop: 12, borderTopWidth: 1, borderTopColor: "#EEE" },
  breakdownSection: { marginBottom: 10 },
  breakdownTitle: { fontSize: 13, fontWeight: "600", color: "#2a9d8f", marginBottom: 6 },
  breakdownItem: { fontSize: 12, color: "#666", marginBottom: 4, marginLeft: 8 },
  emptyText: { textAlign: "center", color: "#999", fontSize: 14, marginVertical: 12 },

  comparisonContainer: { marginTop: 12, paddingTop: 12, borderTopWidth: 1, borderTopColor: "#EEE", backgroundColor: "#f5f5f5", padding: 10, borderRadius: 8 },
  comparisonRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 8 },
  comparisonLabel: { fontSize: 13, color: "#555", fontWeight: "500" },
  comparisonValue: { fontSize: 13, fontWeight: "600", color: "#2a9d8f" },

  progressBarContainer: { height: 8, backgroundColor: "#DDD", borderRadius: 4, marginVertical: 8, overflow: "hidden" },
  progressBar: { height: "100%", borderRadius: 4 },

  energyInfoContainer: { marginBottom: 14, paddingBottom: 12, borderBottomWidth: 1, borderBottomColor: "#EEE", backgroundColor: "#f9f9f9", padding: 10, borderRadius: 8 },
  energyInfoRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 8 },
  energyLabel: { fontSize: 13, color: "#555", fontWeight: "500" },
  energyValue: { fontSize: 13, fontWeight: "600", color: "#2a9d8f" },

  logoutButton: { flexDirection: "row", backgroundColor: "#e63946", padding: 14, borderRadius: 10, justifyContent: "center", alignItems: "center", marginTop: 20, marginBottom: 40 },
  logoutText: { color: "#fff", fontSize: 16, fontWeight: "600", marginLeft: 8 },
});