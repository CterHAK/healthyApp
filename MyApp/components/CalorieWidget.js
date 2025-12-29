import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { getTodayCalories } from "../services/calorieService"; // <-- import service

export default function CalorieWidgetToday({ userData, navigation }) {
  const [loading, setLoading] = useState(true);
  const [todayData, setTodayData] = useState(null);

  useEffect(() => {
    fetchToday();
  }, [userData]);

  const fetchToday = async () => {
    setLoading(true);
    const data = await getTodayCalories(userData?.email, userData?.weight);
    setTodayData(data);
    setLoading(false);
  };

  if (loading) {
    return (
      <View style={styles.card}>
        <Text style={styles.cardTitle}>🔥 Calo Hôm Nay</Text>
        <ActivityIndicator size="small" color="#4ECDC4" />
      </View>
    );
  }

  if (!todayData) {
    return (
      <View style={styles.card}>
        <Text style={styles.cardTitle}>🔥 Calo Hôm Nay</Text>
        <Text style={styles.emptyText}>Chưa có dữ liệu hôm nay</Text>
        <TouchableOpacity style={styles.refreshBtn} onPress={fetchToday}>
          <Ionicons name="refresh" size={16} color="#4ECDC4" />
          <Text style={styles.refreshBtnText}>Cập nhật</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const intake = todayData.intake?.total_calories || 0;
  const burned = todayData.burned?.total_calories || 0;
  const net = todayData.net_calories || 0;

  let netColor = "#FF6B6B";
  if (net >= 0 && net <= 500) netColor = "#4ECDC4";
  if (net > 500) netColor = "#FFA500";

  return (
    <View style={styles.card}>
      <View style={styles.cardHeader}>
        <Text style={styles.cardTitle}>🔥 Calo Hôm Nay</Text>
        <TouchableOpacity onPress={fetchToday}>
          <Ionicons name="refresh" size={18} color="#4ECDC4" />
        </TouchableOpacity>
      </View>

      <View style={styles.calorieRow}>
        <View style={styles.calorieItem}>
          <Text style={styles.calorieLabel}>📥 Nhận vào</Text>
          <Text style={styles.calorieValue}>{Math.round(intake)}</Text>
          <Text style={styles.calorieUnit}>kcal</Text>
        </View>

        <View style={styles.divider} />

        <View style={styles.calorieItem}>
          <Text style={styles.calorieLabel}>🔥 Tiêu thụ</Text>
          <Text style={styles.calorieValue}>{Math.round(burned)}</Text>
          <Text style={styles.calorieUnit}>kcal</Text>
        </View>

        <View style={styles.divider} />

        <View style={styles.calorieItem}>
          <Text style={styles.calorieLabel}>⚖️ Net</Text>
          <Text style={[styles.calorieValue, { color: netColor }]}>
            {Math.round(net)}
          </Text>
          <Text style={styles.calorieUnit}>kcal</Text>
        </View>
      </View>

      {(todayData.intake?.meals?.length > 0 ||
        todayData.burned?.exercises?.length > 0) && (
        <TouchableOpacity
          style={styles.breakdownBtn}
          onPress={() => navigation.navigate("CalorieAnalytics", { userData })}
        >
          <Text style={styles.breakdownBtnText}>📊 Xem chi tiết</Text>
          <Ionicons name="chevron-forward" size={16} color="#4ECDC4" />
        </TouchableOpacity>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: "#FFF",
    borderRadius: 15,
    padding: 16,
    marginBottom: 16,
    shadowColor: "#000",
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  cardHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 12,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: "#333",
  },
  calorieRow: {
    flexDirection: "row",
    justifyContent: "space-around",
    alignItems: "center",
    marginBottom: 12,
  },
  calorieItem: {
    alignItems: "center",
    flex: 1,
  },
  calorieLabel: {
    fontSize: 12,
    color: "#666",
    marginBottom: 4,
  },
  calorieValue: {
    fontSize: 24,
    fontWeight: "700",
    color: "#2a9d8f",
  },
  calorieUnit: {
    fontSize: 11,
    color: "#999",
    marginTop: 2,
  },
  divider: {
    width: 1,
    height: 50,
    backgroundColor: "#EEE",
  },
  breakdownBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#F0F8F7",
    borderRadius: 8,
    paddingVertical: 10,
    marginTop: 8,
  },
  breakdownBtnText: {
    color: "#4ECDC4",
    fontWeight: "600",
    marginRight: 4,
  },
  refreshBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#F0F8F7",
    borderRadius: 8,
    paddingVertical: 8,
    marginTop: 12,
  },
  refreshBtnText: {
    color: "#4ECDC4",
    fontWeight: "600",
    marginLeft: 6,
  },
  emptyText: {
    textAlign: "center",
    color: "#999",
    marginVertical: 12,
  },
});
