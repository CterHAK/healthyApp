// screens/DailyScheduleScreen.js
import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  ScrollView,
  SafeAreaView,
  StatusBar,
  StyleSheet,
  Dimensions,
  ActivityIndicator,
} from "react-native";
import { BarChart, PieChart } from "react-native-chart-kit";
import { getFoodsByEmail } from "../services/food_api";
import { getExercisePlan, getExerciseInfo } from "../services/exercise_api";

// -----------------------------
// Helper format ngày
// -----------------------------
const formatDate = (dateStr) => {
  if (!dateStr) return "Không rõ ngày";
  const d = new Date(dateStr);
  return `${d.getDate()}-${d.getMonth() + 1}-${d.getFullYear()}`;
};

// -----------------------------
// Tính tổng session (tách intake, burned, net)
// -----------------------------
const calculateSessionTotals = (session) => {
  let totals = { intake_cal: 0, burned_cal: 0, net_cal: 0, protein: 0, carb: 0, fat: 0 };
  (session.exercises || []).forEach((ex) => {
    totals.burned_cal += ex.calories_burned || 0;
  });
  (session.meals || []).forEach((meal) => {
    const n = meal.nutrition || {};
    totals.intake_cal += n.calories_kcal || 0;
    totals.protein += n.protein_g || 0;
    totals.carb += n.carbohydrate_g || 0;
    totals.fat += n.fat_g || 0;
  });
  totals.net_cal = totals.intake_cal - totals.burned_cal;
  return totals;
};

// -----------------------------
// Tính phân bổ macros cho PieChart
// -----------------------------
const calculateMacroDistribution = (totals) => {
  const protein_cal = totals.protein * 4;
  const carb_cal = totals.carb * 4;
  const fat_cal = totals.fat * 9;
  const total_macro_cal = protein_cal + carb_cal + fat_cal || 1; // Tránh chia 0
  return [
    { name: 'Protein', cal: protein_cal, color: '#FF6384', legendFontColor: '#7F7F7F', legendFontSize: 12 },
    { name: 'Carb', cal: carb_cal, color: '#36A2EB', legendFontColor: '#7F7F7F', legendFontSize: 12 },
    { name: 'Fat', cal: fat_cal, color: '#FFCE56', legendFontColor: '#7F7F7F', legendFontSize: 12 },
  ];
};

// -----------------------------
// Chuyển đổi dữ liệu thành schedule theo ngày/session
// -----------------------------
const transformDailySchedule = (exercisePlans, foodData) => {
  const schedule = {};

  exercisePlans.forEach((plan) => {
    const date = plan.day?.slice(0, 10) || "Không rõ ngày";
    const session = plan.session || "Khác";
    if (!schedule[date]) schedule[date] = {};
    if (!schedule[date][session]) schedule[date][session] = { exercises: [], meals: [] };
    schedule[date][session].exercises.push(plan);
  });

  foodData.forEach((food) => {
    const date = food.day || "Không rõ ngày";
    const session = food.session || "Khác"; // Sửa typo seassion -> session
    if (!schedule[date]) schedule[date] = {};
    if (!schedule[date][session]) schedule[date][session] = { exercises: [], meals: [] };
    schedule[date][session].meals.push(food);
  });

  return Object.entries(schedule)
    .sort((a, b) => new Date(a[0]) - new Date(b[0]))
    .map(([date, sessions]) => ({ date, sessions }));
};

// -----------------------------
// Lấy calories từ Python backend
// -----------------------------
const fetchCaloriesForExercises = async (exercises) => {
  const updated = await Promise.all(
    exercises.map(async (ex) => {
      try {
        const info = await getExerciseInfo(ex.name);
        if (info.status === "success") {
          const { MET } = info.exercise;
          const sets = ex.sets || 1;
          const reps = ex.reps || 1;
          const weight_kg = 70; // default, có thể lấy từ user
          const totalSeconds = sets * reps * 3; // 3s/rep
          const calories = Math.round(MET * weight_kg * (totalSeconds / 3600) * 100) / 100;
          return { ...ex, calories_burned: calories };
        }
      } catch (e) {
        console.log("Error fetch calories:", ex.name, e.message);
      }
      return { ...ex, calories_burned: ex.calories_burned || 0 };
    })
  );
  return updated;
};

// -----------------------------
// Component
// -----------------------------
export default function DailyScheduleScreen({ route }) {
  const { userData } = route.params || {};
  const [schedule, setSchedule] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const screenWidth = Dimensions.get("window").width;

  useEffect(() => {
    const fetchData = async () => {
      if (!userData?.email) {
        setError("Không có email người dùng");
        setLoading(false);
        return;
      }

      try {
        const exercisesRes = await getExercisePlan(userData.email); // API bài tập
        const foods = await getFoodsByEmail(userData.email); // API bữa ăn

        // Cập nhật calories từ Python backend
        const exercisesWithCalories = await fetchCaloriesForExercises(exercisesRes.results || []);

        const transformed = transformDailySchedule(exercisesWithCalories, foods);
        setSchedule(transformed);
      } catch (err) {
        setError(err.message || "Lỗi khi lấy dữ liệu");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [userData]);

  if (loading)
    return (
      <SafeAreaView style={styles.container}>
        <ActivityIndicator size="large" color="#00796B" />
        <Text style={styles.loadingText}>Đang tải thời khóa biểu...</Text>
      </SafeAreaView>
    );

  if (error)
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.errorText}>{error}</Text>
      </SafeAreaView>
    );

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Text style={styles.mainTitle}>Thời Khóa Biểu Hàng Ngày</Text>
        {schedule.map((dayItem) => {
          // Tổng ngày
          const dailyTotals = Object.values(dayItem.sessions).reduce((acc, session) => {
            const totals = calculateSessionTotals(session);
            acc.intake_cal += totals.intake_cal;
            acc.burned_cal += totals.burned_cal;
            acc.net_cal += totals.net_cal;
            acc.protein += totals.protein;
            acc.carb += totals.carb;
            acc.fat += totals.fat;
            return acc;
          }, { intake_cal: 0, burned_cal: 0, net_cal: 0, protein: 0, carb: 0, fat: 0 });

          const pieData = calculateMacroDistribution(dailyTotals);

          return (
            <View key={dayItem.date} style={styles.dayCard}>
              <Text style={styles.dayTitle}>📅 {formatDate(dayItem.date)}</Text>

              {/* Biểu đồ Bar cho calories và macros */}
              <Text style={styles.chartTitle}>Biểu Đồ Calories & Macros</Text>
              <BarChart
                data={{
                  labels: ["Intake", "Burned", "Net", "Protein", "Carb", "Fat"],
                  datasets: [{ data: [dailyTotals.intake_cal, dailyTotals.burned_cal, dailyTotals.net_cal, dailyTotals.protein, dailyTotals.carb, dailyTotals.fat] }],
                }}
                width={screenWidth - 32}
                height={180}
                chartConfig={{
                  backgroundColor: "#fff",
                  backgroundGradientFrom: "#fff",
                  backgroundGradientTo: "#fff",
                  decimalPlaces: 0,
                  color: (opacity = 1) => `rgba(0, 121, 107, ${opacity})`,
                  labelColor: (opacity = 1) => `rgba(0,0,0,${opacity})`,
                  style: { borderRadius: 12 },
                  propsForBackgroundLines: { stroke: "#eee" },
                }}
                style={{ marginVertical: 10, borderRadius: 12 }}
              />

              {/* Biểu đồ Pie cho phân bổ macros */}
              <Text style={styles.chartTitle}>Phân Bổ Dinh Dưỡng (% Calories)</Text>
              <PieChart
                data={pieData}
                width={screenWidth - 32}
                height={200}
                chartConfig={{
                  color: (opacity = 1) => `rgba(0, 121, 107, ${opacity})`,
                  labelColor: (opacity = 1) => `rgba(0,0,0,${opacity})`,
                }}
                accessor="cal"
                backgroundColor="transparent"
                paddingLeft="15"
                absolute
              />

              {/* Session chi tiết */}
              {Object.entries(dayItem.sessions).map(([sessionName, sessionData]) => {
                const totals = calculateSessionTotals(sessionData);
                return (
                  <View key={sessionName} style={styles.sessionCard}>
                    <Text style={styles.sessionTitle}>🕑 Session: {sessionName}</Text>

                    {/* Lịch tập luyện */}
                    <Text style={styles.subTitle}>🏋️‍♂️ Lịch Tập Luyện:</Text>
                    {sessionData.exercises.length > 0 ? (
                      sessionData.exercises.map((ex) => (
                        <Text key={ex._id} style={styles.itemText}>
                          - {ex.name}: {ex.sets} sets × {ex.reps} reps | Đốt {ex.calories_burned || 0} kcal
                        </Text>
                      ))
                    ) : (
                      <Text style={styles.itemText}>Không có bài tập</Text>
                    )}

                    {/* Lịch ăn uống */}
                    <Text style={styles.subTitle}>🍽️ Lịch Ăn Uống:</Text>
                    {sessionData.meals.length > 0 ? (
                      sessionData.meals.map((meal, idx) => (
                        <Text key={idx} style={styles.itemText}>
                          - {meal.dish_name}: {meal.nutrition?.calories_kcal || 0} kcal | Protein: {meal.nutrition?.protein_g || 0}g | Carb: {meal.nutrition?.carbohydrate_g || 0}g | Fat: {meal.nutrition?.fat_g || 0}g
                        </Text>
                      ))
                    ) : (
                      <Text style={styles.itemText}>Không có bữa ăn</Text>
                    )}

                    {/* Tổng session */}
                    <Text style={styles.sessionTotal}>
                      Tổng Session: Intake {totals.intake_cal} kcal | Burned {totals.burned_cal} kcal | Net {totals.net_cal} kcal | {totals.protein}g P | {totals.carb}g C | {totals.fat}g F
                    </Text>
                  </View>
                );
              })}

              {/* Tổng ngày */}
              <Text style={styles.dayTotal}>
                Tổng Ngày: Intake {dailyTotals.intake_cal} kcal | Burned {dailyTotals.burned_cal} kcal | Net {dailyTotals.net_cal} kcal | {dailyTotals.protein}g P | {dailyTotals.carb}g C | {dailyTotals.fat}g F
              </Text>
            </View>
          );
        })}
      </ScrollView>
    </SafeAreaView>
  );
}

// =========================
// Styles (cập nhật thêm)
// =========================
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#FAFAFA" },
  scrollContent: { paddingHorizontal: 16, paddingBottom: 40 },
  mainTitle: { fontSize: 22, fontWeight: "bold", textAlign: "center", marginVertical: 16, color: "#00796B" },
  loadingText: { marginTop: 20, textAlign: "center", fontSize: 16 },
  errorText: { marginTop: 20, textAlign: "center", fontSize: 16, color: "red" },
  dayCard: {
    backgroundColor: "#FFF",
    borderRadius: 15,
    padding: 12,
    marginBottom: 20,
    shadowColor: "#000",
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  dayTitle: { fontSize: 18, fontWeight: "700", color: "#37474F", marginBottom: 6 },
  chartTitle: { fontSize: 16, fontWeight: "600", color: "#00796B", marginTop: 10, marginBottom: 4 },
  sessionCard: { padding: 8, borderTopWidth: 1, borderTopColor: "#EEE", marginTop: 8 },
  sessionTitle: { fontSize: 16, fontWeight: "600", color: "#00796B", marginBottom: 4 },
  subTitle: { fontSize: 15, fontWeight: "500", color: "#555", marginTop: 6, marginBottom: 2 },
  itemText: { fontSize: 14, color: "#555", marginVertical: 2 },
  sessionTotal: { fontSize: 14, fontWeight: "700", marginTop: 6, color: "#000" },
  dayTotal: { fontSize: 15, fontWeight: "bold", marginTop: 12, color: "#D32F2F", textAlign: "center" },
});