// App.js
import React from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createStackNavigator } from "@react-navigation/stack";

// 👉 Các màn hình gốc
import LoginScreen from "./screens/LoginScreen";
import RegisterScreen from "./screens/RegisterScreen";
import HomeScreen from "./screens/HomeScreen"; // Bottom Tab Navigator
import FoodHistoryScreen from "./screens/FoodHistoryScreen";

// 👉 Các màn hình Tập luyện
import ChatBotScreen from "./screens/ChatBotScreen";
import PlanByDateScreen from "./screens/PlanByDateScreen";
import ExerciseGroupAddScreen from "./screens/ExerciseGroupAddScreen";
const Stack = createStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="Login"
        screenOptions={{ headerShown: false }}
      >
        {/* Đăng nhập / đăng ký */}
        <Stack.Screen name="Login" component={LoginScreen} />
        <Stack.Screen name="Register" component={RegisterScreen} />

        {/* Trang chính (Bottom Tab) */}
        <Stack.Screen name="Home" component={HomeScreen} />

        {/* Lịch sử ăn uống */}
        <Stack.Screen name="FoodHistory" component={FoodHistoryScreen} />

        {/* 💪 Các trang tập luyện */}
        <Stack.Screen name="ExerciseGroupAdd" component={ExerciseGroupAddScreen} />
        <Stack.Screen name="PlanByDate" component={PlanByDateScreen} /> 

      </Stack.Navigator>
    </NavigationContainer>
  );
}
