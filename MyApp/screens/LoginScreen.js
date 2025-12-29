import React, { useState } from "react";
import { View, Text, TextInput, Button, StyleSheet, Alert } from "react-native";
import { loginAccount, getAccountByEmail } from "../services/account_api";
import User from "../models/User.js"; // ✅ import class User
import { getUserByEmail } from "../services/user_api.js";
export default function LoginScreen({ navigation }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [user, setUser] = useState(new User()); // ✅ dùng model User

  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert("Thiếu thông tin", "Vui lòng nhập đầy đủ email và mật khẩu.");
      return;
    }

  

try {
  // ✅ Gọi API đăng nhập
  setLoading(true);
  await loginAccount(email, password);
  console.log("Login success");

  // ✅ Lấy thông tin người dùng
  const userDataFromAPI = await getUserByEmail(email);
  console.log("Fetched user data:", userDataFromAPI);

  // ✅ Chuẩn hóa dữ liệu để frontend sử dụng
  const loggedInUser = {
    ...userDataFromAPI,
    caloriePlan: Number(userDataFromAPI.caloriePlan) || 0,
    bmr: Number(userDataFromAPI.bmr) || 0,
    tdee: Number(userDataFromAPI.tdee) || 0,
    allergies: Array.isArray(userDataFromAPI.allergies) ? userDataFromAPI.allergies : [],
    diseases: Array.isArray(userDataFromAPI.diseases) ? userDataFromAPI.diseases : [],
  };

  setUser(loggedInUser);

  // ✅ Chuyển đến trang Home
  navigation.replace("Home", { userData: loggedInUser });
} catch (err) {
  console.error(err);
  Alert.alert("Đăng nhập thất bại", "Email hoặc mật khẩu không đúng.");
} finally {
  setLoading(false);
}

  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Đăng nhập</Text>

      <TextInput
        style={styles.input}
        placeholder="Email"
        keyboardType="email-address"
        value={email}
        onChangeText={setEmail}
      />

      <TextInput
        style={styles.input}
        placeholder="Mật khẩu"
        secureTextEntry
        value={password}
        onChangeText={setPassword}
      />

      <Button
        title={loading ? "Đang đăng nhập..." : "Đăng nhập"}
        onPress={handleLogin}
        disabled={loading}
      />

      <View style={{ marginTop: 20 }}>
        <Button
          title="Tạo tài khoản mới"
          onPress={() => navigation.navigate("Register")}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    padding: 20,
    backgroundColor: "#fff",
  },
  title: {
    fontSize: 26,
    fontWeight: "bold",
    marginBottom: 25,
    textAlign: "center",
    color: "#333",
  },
  input: {
    borderWidth: 1,
    borderColor: "#ccc",
    padding: 12,
    borderRadius: 8,
    marginBottom: 15,
  },
});
