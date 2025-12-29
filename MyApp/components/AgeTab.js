import React, { useRef, useState } from "react";
import { View, Text, FlatList, StyleSheet, Dimensions } from "react-native";

const ITEM_HEIGHT = 50; // chiều cao mỗi item
const PICKER_HEIGHT = ITEM_HEIGHT * 5; // hiển thị 5 item cùng lúc

export default function AgePicker({ min = 1, max = 100, onSelect }) {
  const ages = Array.from({ length: max - min + 1 }, (_, i) => min + i);
  const flatListRef = useRef(null);
  const [selectedIndex, setSelectedIndex] = useState(Math.floor(ages.length / 2));

  const handleScroll = (event) => {
    const yOffset = event.nativeEvent.contentOffset.y;
    const index = Math.round(yOffset / ITEM_HEIGHT);
    setSelectedIndex(index);
    if (onSelect) onSelect(ages[index]);
  };

  const renderItem = ({ item, index }) => {
    const isSelected = index === selectedIndex;
    return (
      <View style={[styles.itemContainer, isSelected && styles.selectedItem]}>
        <Text style={[styles.itemText, isSelected && styles.selectedText]}>
          {item}
        </Text>
      </View>
    );
  };

  return (
    <View style={styles.container}>
      {/* Overlay highlight nằm giữa picker */}
      <View
        style={[
          styles.overlay,
          { top: (PICKER_HEIGHT / 2) - (ITEM_HEIGHT / 2) },
        ]}
      />
      <FlatList
        ref={flatListRef}
        data={ages}
        keyExtractor={(item) => item.toString()}
        renderItem={renderItem}
        showsVerticalScrollIndicator={false}
        snapToInterval={ITEM_HEIGHT}
        decelerationRate="fast"
        onScroll={handleScroll}
        scrollEventThrottle={16}
        contentContainerStyle={{
          paddingTop: PICKER_HEIGHT / 2 - ITEM_HEIGHT / 2,
          paddingBottom: PICKER_HEIGHT / 2 - ITEM_HEIGHT / 2,
        }}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    height: PICKER_HEIGHT, // giới hạn picker cao 5 item
    width: 100,
    alignSelf: "center",
    justifyContent: "center",
  },
  itemContainer: {
    height: ITEM_HEIGHT,
    justifyContent: "center",
    alignItems: "center",
  },
  itemText: {
    fontSize: 20,
    color: "#777",
  },
  selectedText: {
    fontSize: 28,
    color: "#e53935",
    fontWeight: "bold",
  },
  overlay: {
    position: "absolute",
    height: ITEM_HEIGHT,
    width: "100%",
    borderTopWidth: 2,
    borderBottomWidth: 2,
    borderColor: "#e53935",
    zIndex: 1,
  },
});
