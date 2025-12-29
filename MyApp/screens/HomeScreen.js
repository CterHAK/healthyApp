// HomeScreen.js
import React, { useState, useEffect } from "react";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { Ionicons } from "@expo/vector-icons";

// Màn hình con
import Dashboard from "./DashBoardScreen";
import FoodCapture from "./FoodCaptureScreen";
import DailyScheduleScreen from "./DailyScheduleScreen";
import ExerciseGroupAddScreen from "./ExerciseGroupAddScreen";
import PlanByDateScreen from './PlanByDateScreen'
import ChatBotScreen from "./ChatBotScreen";

function Profile() { return null; }
function Settings() { return null; }

const Tab = createBottomTabNavigator();

export default function HomeScreen({ route, navigation }) {
  const userData = route.params?.userData || {};
  const selectedTabFromParams = route.params?.selectedTab || "Dashboard";
  const [currentTab, setCurrentTab] = useState(selectedTabFromParams);

  useEffect(() => {
    if (selectedTabFromParams && selectedTabFromParams !== currentTab) {
      setCurrentTab(selectedTabFromParams);
    }
  }, [selectedTabFromParams]);

  return (
    <Tab.Navigator
      initialRouteName={currentTab}
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarStyle: { backgroundColor: "#fff", height: 70 },
        tabBarLabelStyle: { fontSize: 13, marginBottom: 5 },
        tabBarIcon: ({ color }) => {
          let iconName;
          if (route.name === "Dashboard") iconName = "stats-chart";
          else if (route.name === "Profile") iconName = "person";
          else if (route.name === "Settings") iconName = "settings";
          else if (route.name === "FoodCapture") iconName = "camera";
          else if (route.name === "ChatBot") iconName = "chatbubbles";
          else if (route.name === "PlanByDate") iconName = "barbell";
          return <Ionicons name={iconName} size={22} color={color} />;
        },
        tabBarActiveTintColor: "#43AA8B",
        tabBarInactiveTintColor: "#999",
      })}
    >
      <Tab.Screen
        name="Dashboard"
        component={Dashboard}
        initialParams={{ userData }}
      />
      <Tab.Screen
        name="FoodCapture"
        component={FoodCapture}
        initialParams={{ userData }}
      />
      <Tab.Screen
        name="ChatBot"
        component={ChatBotScreen}
        initialParams={{ userData }}
      />
      <Tab.Screen
        name="PlanByDate"
        component={PlanByDateScreen}
        initialParams={{ userData }}
      />
    </Tab.Navigator>
  );
}