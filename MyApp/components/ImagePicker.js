import React from "react";
import { TouchableOpacity, Text, Image, StyleSheet } from "react-native";

export default function ImagePicker({ avatar, onPick }) {
  return (
    <TouchableOpacity style={styles.imagePicker} onPress={onPick}>
      {avatar ? (
        <Image source={{ uri: avatar }} style={styles.avatar} />
      ) : (
        <Text>Chọn ảnh đại diện</Text>
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  imagePicker: { width: 100, height: 100, borderWidth: 1, borderColor: "#ccc", borderRadius: 8, justifyContent: "center", alignItems: "center", marginBottom: 20 },
  avatar: { width: 100, height: 100, borderRadius: 8 },
});