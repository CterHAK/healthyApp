// services/cloudinary_api.js

import { Platform } from "react-native";

const CLOUD_NAME = "da4f83k0v";
const UPLOAD_PRESET = "save_image_url";

export const uploadImageToCloudinary = async (imageUri) => {
  if (!imageUri) return null;

  try {
    const formData = new FormData();
    const fileUri = Platform.OS === "ios" ? imageUri.replace("file://", "") : imageUri;

    formData.append("file", {
      uri: fileUri,
      type: "image/jpeg",
      name: `food_${Date.now()}.jpg`,
    });

    formData.append("upload_preset", UPLOAD_PRESET);
    formData.append("folder", "FoodHistory");

    const response = await fetch(`https://api.cloudinary.com/v1_1/${CLOUD_NAME}/image/upload`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok || !data.secure_url) {
      throw new Error(data.error?.message || "Upload thất bại");
    }

    return data.secure_url;
  } catch (err) {
    console.error("❌ Cloudinary Upload Error:", err);
    throw err;
  }
};
