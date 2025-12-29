import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  Image,
  StyleSheet,
  TouchableOpacity,
  SafeAreaView,
  ScrollView,
  StatusBar,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { getAllFoods  } from "../services/food_api";

// 🔹 Hàm format ngày hiển thị (hỗ trợ cả "Thứ 3" hoặc "27-10-2025")
const formatDate = (dateStr) => {
  if (!dateStr) return "Không rõ ngày";

  // Nếu là kiểu "Thứ 3"
  if (dateStr.startsWith("Thứ")) return dateStr;

  // Nếu là kiểu "27-10-2025"
  const parts = dateStr.split("-");
  if (parts.length === 3) return `${parts[0]}/${parts[1]}/${parts[2]}`;
  return dateStr;
};

// 🔹 Chuyển dữ liệu API thành định dạng giao diện
const transformApiData = (apiData) => {
  const groupedByDate = apiData.reduce((acc, item) => {
    const date = item.day || "Không rõ ngày";
    const mealType = item.seassion || "Khác";

    // ⚙️ Dữ liệu nutrition là object, không phải chuỗi
    const nutrition = item.nutrition || {
      calories_kcal: 0,
      protein_g: 0,
      carbohydrate_g: 0,
      fat_g: 0,
    };

    const food = {
      name: item.dish_name,
      cal: nutrition.calories_kcal || 0,
      protein: nutrition.protein_g || 0,
      carb: nutrition.carbohydrate_g || 0,
      fat: nutrition.fat_g || 0,
      image_url: item.image_url,
    };

    let dateGroup = acc.find((group) => group.date === date);
    if (!dateGroup) {
      dateGroup = { date, meals: {} };
      acc.push(dateGroup);
    }

    if (!dateGroup.meals[mealType]) {
      dateGroup.meals[mealType] = [];
    }

    dateGroup.meals[mealType].push(food);

    return acc;
  }, []);

  return groupedByDate;
};

// 🔹 Tính tổng dinh dưỡng trong ngày
const calculateDailyTotal = (meals) => {
  let total = { cal: 0, protein: 0, carb: 0, fat: 0 };
  Object.values(meals).forEach((mealList) => {
    mealList.forEach((item) => {
      total.cal += item.cal || 0;
      total.protein += item.protein || 0;
      total.carb += item.carb || 0;
      total.fat += item.fat || 0;
    });
  });
  return total;
};

// 🔹 Component chính
export default function FoodHistoryScreen({ route, navigation }) {
  const { userData } = route.params || {};
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

useEffect(() => {
  const fetchFoods = async () => {
    if (!userData?.email) {
      setError("Không có email người dùng");
      setLoading(false);
      return;
    }

    try {
      const res = await getAllFoods(userData.email); // ✅ đúng email

      if (res.status === "success" && Array.isArray(res.data)) {
        const totalCalories = res.data.reduce(
          (sum, item) => sum + (item.nutrition?.calories_kcal || 0),
          0
        );
        console.log("Total calories:", totalCalories);

        const transformedData = transformApiData(res.data); // ✅ dùng res.data
        setData(transformedData);
      } else {
        console.warn("No foods or invalid response", res);
        setData([]);
      }
    } catch (err) {
      console.error("Fetch foods error:", err);
      setError(err.message || "Lỗi khi lấy dữ liệu");
    } finally {
      setLoading(false);
    }
  };

  fetchFoods();
}, [userData]);

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.loadingText}>Đang tải...</Text>
      </SafeAreaView>
    );
  }

  if (error) {
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.errorText}>Lỗi: {error}</Text>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />
      <View style={styles.headerBar}>
        <TouchableOpacity
          onPress={() => navigation.goBack()}
          style={styles.backButton}
        >
          <Ionicons name="arrow-back" size={24} color="#333" />
        </TouchableOpacity>
        <Text style={styles.screenTitle}>📖 Lịch sử ăn uống</Text>
        <View style={{ width: 30 }} />
      </View>

      <ScrollView contentContainerStyle={styles.scrollContent}>
        {data.map((item) => {
          const total = calculateDailyTotal(item.meals);
          return (
            <View key={item.date} style={styles.dateGroupContainer}>
              <View style={styles.dateHeader}>
                <Text style={styles.dateTitle}>📅 {formatDate(item.date)}</Text>
                <Text style={styles.nutritionText}>
                  🔥 {total.cal} kcal | 🥩 {total.protein}g P | 🍚 {total.carb}g C | 🧈 {total.fat}g F
                </Text>
              </View>

              {Object.entries(item.meals).map(([mealType, foods]) => (
                <View key={mealType} style={styles.mealSection}>
                  <Text style={styles.mealTitle}>🍽️ {mealType}</Text>
                  {foods.map((food, index) => (
                    <View key={index} style={styles.foodCard}>
                      <Image
                        source={{
                          uri:
                            food.image_url ||
                            "https://cdn-icons-png.flaticon.com/512/3075/3075977.png",
                        }}
                        style={styles.foodImage}
                      />
                      <View style={styles.foodInfo}>
                        <Text style={styles.foodName}>{food.name}</Text>
                        <Text style={styles.foodNutrient}>
                          🔥 {food.cal} kcal | 🥩 {food.protein}g | 🍚 {food.carb}g | 🧈 {food.fat}g
                        </Text>
                      </View>
                    </View>
                  ))}
                </View>
              ))}
            </View>
          );
        })}
      </ScrollView>
    </SafeAreaView>
  );
}

// =========================
// 🔹 StyleSheet
// =========================
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#FAFAFA" },
  headerBar: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 16,
    paddingVertical: 10,
    backgroundColor: "#FFF",
    borderBottomWidth: 1,
    borderBottomColor: "#EEE",
  },
  backButton: { padding: 6 },
  screenTitle: {
    fontSize: 20,
    fontWeight: "700",
    color: "#333",
    flex: 1,
    textAlign: "center",
  },
  scrollContent: { paddingHorizontal: 16, paddingBottom: 100 },
  dateGroupContainer: {
    marginBottom: 20,
    backgroundColor: "#FFF",
    borderRadius: 15,
    padding: 12,
    shadowColor: "#000",
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  dateHeader: {
    borderBottomWidth: 1,
    borderBottomColor: "#EEE",
    paddingBottom: 6,
    marginBottom: 8,
  },
  dateTitle: { fontSize: 18, fontWeight: "700", color: "#37474F" },
  nutritionText: { fontSize: 14, color: "#555", marginTop: 4 },
  mealSection: { marginBottom: 10 },
  mealTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#00796B",
    marginBottom: 6,
  },
  foodCard: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#F5F5F5",
    borderRadius: 10,
    padding: 8,
    marginBottom: 6,
  },
  foodImage: { width: 45, height: 45, marginRight: 10 },
  foodInfo: { flex: 1 },
  foodName: { fontSize: 15, fontWeight: "600" },
  foodNutrient: { fontSize: 13, color: "#666" },
  loadingText: { fontSize: 18, textAlign: "center", marginTop: 20 },
  errorText: {
    fontSize: 18,
    color: "red",
    textAlign: "center",
    marginTop: 20,
  },
});
