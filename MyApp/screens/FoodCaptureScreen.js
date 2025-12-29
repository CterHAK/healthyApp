// screens/FoodCaptureScreen.js
import React, { useState } from "react";
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  TextInput,
  Alert,
  ActivityIndicator,
  Dimensions,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { ProgressChart } from "react-native-chart-kit";
import { saveFood } from "../services/food_api";
import { recognizeFood } from "../services/food_recognition_api";
import { uploadImageToCloudinary } from "../services/cloudinary_api";

const screenWidth = Dimensions.get("window").width;

export default function FoodCaptureScreen({ route, navigation }) {
  const { userData } = route.params || {};
  const [image, setImage] = useState(null);
  const [nutrition, setNutrition] = useState({ calories: "", protein: "", carbs: "", fat: "" });
  const [mealType, setMealType] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [dishName, setDishName] = useState("");
  const [ingredients, setIngredients] = useState([]);
  const [newIngredient, setNewIngredient] = useState("");
  const [portionSize, setPortionSize] = useState("");
  const [day, setDay] = useState("");

  const mealOptions = ["Sáng", "Trưa", "Tối", "Bữa nhẹ"];

  // ------------------- PICK IMAGE -------------------
  const pickImage = async () => {
    try {
      const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!permissionResult.granted) {
        Alert.alert("Lỗi", "Bạn cần cấp quyền truy cập thư viện ảnh!");
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.5,
      });

      if (!result.canceled) {
        setImage(result.assets[0].uri);
      }
    } catch (error) {
      console.error("❌ Error picking image:", error);
      Alert.alert("Lỗi", "Không thể chọn ảnh: " + error.message);
    }
  };

  // ------------------- INGREDIENTS -------------------
  const addIngredient = () => {
    if (newIngredient.trim()) {
      setIngredients([...ingredients, newIngredient.trim()]);
      setNewIngredient("");
    } else {
      Alert.alert("⚠️ Lỗi", "Vui lòng nhập nguyên liệu!");
    }
  };

  const removeIngredient = (index) => {
    setIngredients(ingredients.filter((_, i) => i !== index));
  };

  // ------------------- NUTRITION -------------------
  const updateNutrition = (field, value) => {
    const numericValue = value.replace(/[^0-9.]/g, "").replace(/(\..*?)\./g, "$1");
    setNutrition(prev => ({ ...prev, [field]: numericValue }));
  };

  // ------------------- SAVE TO BACKEND -------------------
  const saveToBackend = async () => {
    if (!image || !dishName || !mealType || !nutrition.calories || !nutrition.protein || !nutrition.carbs || !nutrition.fat) {
      Alert.alert("⚠️ Thiếu thông tin", "Vui lòng nhập đầy đủ tên món ăn, loại bữa ăn và dinh dưỡng.");
      return;
    }

    try {
      setIsUploading(true);
      const imageUrl = await uploadImageToCloudinary(image);
      if (!imageUrl) return;

      const foodData = {
        email: userData?.email || "",
        dish_name: dishName,
        ingredients: ingredients.length ? ingredients : ["Không có nguyên liệu"],
        portion_size: portionSize || "Không có thông tin",
        nutrition: {
          calories_kcal: parseFloat(nutrition.calories) || 0,
          protein_g: parseFloat(nutrition.protein) || 0,
          carbohydrate_g: parseFloat(nutrition.carbs) || 0,
          fat_g: parseFloat(nutrition.fat) || 0,
        },
        image_url: imageUrl,
        day: day || new Date().toLocaleDateString("vi-VN").replace(/\//g, "-"),
        session: mealType,
        is_recognized: true,
      };

      const response = await saveFood(foodData);
      console.log("[DEBUG] Backend response:", response);
      Alert.alert("✅ Lưu thành công!", `${dishName} đã được lưu!`);
      resetForm();
    } catch (error) {
      console.error("❌ Save error:", error);
      Alert.alert("❌ Lỗi", "Không thể lưu dữ liệu: " + error.message);
    } finally {
      setIsUploading(false);
    }
  };

  const resetForm = () => {
    setImage(null);
    setNutrition({ calories: "", protein: "", carbs: "", fat: "" });
    setMealType(null);
    setDishName("");
    setIngredients([]);
    setNewIngredient("");
    setPortionSize("");
    setDay("");
  };

  // ------------------- AUTO RECOGNITION -------------------
  const autoRecognizeFood = async () => {
    if (!image) {
      Alert.alert("⚠️ Lỗi", "Vui lòng chọn ảnh trước khi nhận diện");
      return;
    }

    try {
      setIsRecognizing(true);
      const imageUrl = await uploadImageToCloudinary(image);
      if (!imageUrl) return;

      const result = await recognizeFood(imageUrl);

      if (result?.status === "success") {
        const foodInfo = result.info || {};
        setDishName(foodInfo.dish_name || "");
        
        // Parse ingredients if it's a JSON string
        let parsedIngredients = foodInfo.ingredients || [];
        if (typeof parsedIngredients === "string") {
          try {
            parsedIngredients = JSON.parse(parsedIngredients);
          } catch (e) {
            console.warn("Could not parse ingredients:", e);
            parsedIngredients = [];
          }
        }
        setIngredients(Array.isArray(parsedIngredients) ? parsedIngredients : []);
        
        // Parse portion_size if it's a JSON string
        let parsedPortionSize = foodInfo.portion_size || "";
        if (typeof parsedPortionSize === "string" && parsedPortionSize.startsWith("[")) {
          try {
            const portions = JSON.parse(parsedPortionSize);
            parsedPortionSize = Array.isArray(portions) ? portions.join(", ") : parsedPortionSize;
          } catch (e) {
            console.warn("Could not parse portion_size:", e);
          }
        }
        setPortionSize(parsedPortionSize);
        
        // Format nutrition data from API response
        const nutritionData = foodInfo.nutrition || {};
        setNutrition({
          calories: nutritionData.calories_kcal?.toString() || nutritionData.calories?.toString() || "",
          protein: nutritionData.protein_g?.toString() || nutritionData.protein?.toString() || "",
          carbs: nutritionData.carbohydrate_g?.toString() || nutritionData.carbs?.toString() || "",
          fat: nutritionData.fat_g?.toString() || nutritionData.fat?.toString() || "",
        });

        Alert.alert(
          "✅ Nhận diện thành công!",
          `🍽️ ${foodInfo.dish_name || "Món ăn"}\nVui lòng kiểm tra thông tin và chọn bữa ăn để lưu`
        );
      } else {
        Alert.alert("❌ Lỗi nhận diện", result?.message || "Không thể nhận diện món ăn");
      }
    } catch (error) {
      console.error("❌ Recognition error:", error);
      Alert.alert("❌ Lỗi", "Không thể nhận diện: " + error.message);
    } finally {
      setIsRecognizing(false);
    }
  };

  // ================= RENDER =================
  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {/* Header */}
      <View style={styles.headerContainer}>
        <Text style={styles.headerTitle}>Nhận diện món ăn</Text>
        <TouchableOpacity
          style={styles.historyButtonNew}
          onPress={() => navigation.navigate("FoodHistory", { userData })}
        >
          <Text style={styles.historyButtonTextNew}>Lịch sử</Text>
        </TouchableOpacity>
      </View>

      {/* Upload Box */}
      <TouchableOpacity onPress={pickImage} style={styles.uploadBox}>
        {image ? <Image source={{ uri: image }} style={styles.image} /> :
          <View style={styles.uploadPlaceholder}>
            <Text style={styles.uploadText}>📷</Text>
            <Text style={styles.uploadSubText}>Chọn hoặc chụp ảnh món ăn</Text>
          </View>}
      </TouchableOpacity>

      {/* Auto Recognition Button */}
      {image && (
        <TouchableOpacity style={[styles.mainButton, isRecognizing && styles.buttonLoading]} onPress={autoRecognizeFood} disabled={isRecognizing}>
          {isRecognizing ? <ActivityIndicator color="#fff" /> : <Text style={styles.mainButtonText}>Nhận diện tự động</Text>}
        </TouchableOpacity>
      )}

      {/* Dish Name */}
      <TextInput style={styles.input} placeholder="Tên món ăn" value={dishName} onChangeText={setDishName} autoCapitalize="words" />

      {/* Ingredients */}
      <Text style={styles.inputLabel}>Nguyên liệu</Text>
      <View style={styles.ingredientInputContainer}>
        <TextInput style={[styles.input, styles.ingredientInput]} placeholder="Nhập nguyên liệu mới" value={newIngredient} onChangeText={setNewIngredient} />
        <TouchableOpacity onPress={addIngredient} style={styles.addButton}><Text style={styles.addButtonText}>Thêm</Text></TouchableOpacity>
      </View>
      {ingredients.length > 0 && (
        <View style={styles.ingredientList}>
          {ingredients.map((i, idx) => (
            <View key={idx} style={styles.ingredientItem}>
              <Text style={styles.ingredientText}>{i}</Text>
              <TouchableOpacity onPress={() => removeIngredient(idx)}><Text style={styles.removeButtonText}>🗑️</Text></TouchableOpacity>
            </View>
          ))}
        </View>
      )}

      {/* Day */}
      <TextInput style={styles.input} placeholder="Ngày (ví dụ: 1-1-2025)" value={day} onChangeText={setDay} keyboardType="numeric" />

      {/* Meal Type */}
      <Text style={styles.mealLabel}>Chọn loại bữa ăn:</Text>
      <View style={styles.mealOptions}>
        {mealOptions.map(option => (
          <TouchableOpacity key={option} onPress={() => setMealType(option)} style={[styles.mealButton, mealType === option && styles.mealButtonActive]}>
            <Text style={[styles.mealButtonText, mealType === option && styles.mealButtonTextActive]}>{option}</Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Nutrition Chart */}
      {nutrition.calories && nutrition.protein && nutrition.carbs && nutrition.fat && (
        <View style={styles.card}>
          <View style={styles.foodHeader}>
            <Text style={styles.foodName}>{dishName || "Món ăn"}</Text>
            <Text style={styles.calories}>{nutrition.calories} kcal</Text>
          </View>
          <ProgressChart
            data={{ labels: ["Protein", "Carbs", "Fat"], data: [Math.min(parseFloat(nutrition.protein)/100,1), Math.min(parseFloat(nutrition.carbs)/100,1), Math.min(parseFloat(nutrition.fat)/100,1)] }}
            width={screenWidth - 40} height={220} strokeWidth={16} radius={50}
            chartConfig={{ backgroundGradientFrom: "#fff", backgroundGradientTo: "#fff", color: (opacity = 1) => `rgba(21,101,192,${opacity})` }}
            style={styles.chart} hideLegend={false}
          />
        </View>
      )}

      {/* Save Button */}
      <TouchableOpacity style={[styles.mainButton, (!image || !dishName || !mealType || !nutrition.calories || !nutrition.protein || !nutrition.carbs || !nutrition.fat) && styles.buttonDisabled, isUploading && styles.buttonLoading]} onPress={saveToBackend} disabled={isUploading || !image || !dishName || !mealType || !nutrition.calories || !nutrition.protein || !nutrition.carbs || !nutrition.fat}>
        {isUploading ? <ActivityIndicator color="#fff" /> : <Text style={styles.mainButtonText}>Lưu món ăn</Text>}
      </TouchableOpacity>

      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

// ================= STYLES =================
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f1f5f9" },
  headerContainer: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", paddingHorizontal: 16, marginVertical: 12 },
  headerTitle: { fontSize: 24, fontWeight: "bold", color: "#1565c0" },
  historyButtonNew: { backgroundColor: "#bbdefb", paddingVertical: 10, paddingHorizontal: 16, borderRadius: 20 },
  historyButtonTextNew: { color: "#1565c0", fontWeight: "bold", fontSize: 16 },
  uploadBox: { margin: 16, height: 220, borderRadius: 16, borderWidth: 2, borderColor: "#1565c0", borderStyle: "dashed", justifyContent: "center", alignItems: "center", backgroundColor: "#fff" },
  uploadPlaceholder: { alignItems: "center" },
  uploadText: { fontSize: 48, marginBottom: 8 },
  uploadSubText: { color: "#777", fontSize: 16, textAlign: "center" },
  image: { width: "100%", height: "100%", borderRadius: 14 },
  mainButton: { backgroundColor: "#1565c0", paddingVertical: 14, borderRadius: 25, alignItems: "center", marginHorizontal: 16, marginBottom: 20 },
  buttonLoading: { backgroundColor: "#0d47a1" },
  buttonDisabled: { backgroundColor: "#90a4ae" },
  mainButtonText: { color: "#fff", fontWeight: "bold", fontSize: 16 },
  input: { backgroundColor: "#fff", marginHorizontal: 16, borderRadius: 12, paddingHorizontal: 14, paddingVertical: 12, marginBottom: 12, fontSize: 16 },
  inputLabel: { marginHorizontal: 16, fontSize: 18, fontWeight: "bold", color: "#1565c0", marginBottom: 8 },
  ingredientInputContainer: { flexDirection: "row", marginHorizontal: 16, marginBottom: 12 },
  ingredientInput: { flex: 1, marginRight: 10 },
  addButton: { backgroundColor: "#1565c0", paddingVertical: 12, paddingHorizontal: 16, borderRadius: 12 },
  addButtonText: { color: "#fff", fontWeight: "bold", fontSize: 14 },
  ingredientList: { marginHorizontal: 16, marginBottom: 20 },
  ingredientItem: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", backgroundColor: "#e3f2fd", padding: 10, borderRadius: 8, marginBottom: 6 },
  ingredientText: { fontSize: 16, color: "#1565c0" },
  removeButtonText: { fontSize: 18, color: "#e53935" },
  mealLabel: { marginHorizontal: 16, fontSize: 18, fontWeight: "bold", color: "#1565c0", marginBottom: 8 },
  mealOptions: { flexDirection: "row", flexWrap: "wrap", justifyContent: "space-around", marginBottom: 20 },
  mealButton: { borderWidth: 2, borderColor: "#1565c0", borderRadius: 25, paddingVertical: 12, paddingHorizontal: 20, margin: 4 },
  mealButtonActive: { backgroundColor: "#1565c0" },
  mealButtonText: { color: "#1565c0", fontWeight: "600", fontSize: 14 },
  mealButtonTextActive: { color: "#fff", fontWeight: "700" },
  card: { backgroundColor: "#fff", borderRadius: 16, padding: 20, marginHorizontal: 16, marginBottom: 20 },
  foodHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 16 },
  foodName: { fontSize: 22, fontWeight: "bold", color: "#1565c0" },
  calories: { color: "#e53935", fontSize: 18, fontWeight: "bold" },
  chart: { marginVertical: 16, borderRadius: 12 },
});
