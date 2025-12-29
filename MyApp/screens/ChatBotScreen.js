// ChatBotScreen.js
import React, { useState, useEffect, useRef } from "react";
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TextInput,
  TouchableOpacity,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  SafeAreaView,
  Alert,
} from "react-native";
import { Ionicons, MaterialCommunityIcons } from "@expo/vector-icons";
import AsyncStorage from '@react-native-async-storage/async-storage';
import MarkdownDisplay from 'react-native-markdown-display';
import { sendChatMessage } from "../services/chat_api";

const CHAT_HISTORY_KEY = "@food_chatbot_history";
const MAX_CHAT_HISTORY = 100;

// ==================== MARKDOWN STYLES ====================
const markdownStyles = {
  body: {
    color: '#333',
    fontSize: 14,
    lineHeight: 20,
  },
  heading1: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginVertical: 8,
  },
  heading2: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginVertical: 6,
  },
  heading3: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginVertical: 4,
  },
  heading4: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#333',
    marginVertical: 4,
  },
  heading5: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
    marginVertical: 4,
  },
  heading6: {
    fontSize: 13,
    fontWeight: 'bold',
    color: '#666',
    marginVertical: 4,
  },
  em: {
    fontStyle: 'italic',
    color: '#555',
  },
  strong: {
    fontWeight: 'bold',
    color: '#333',
  },
  paragraph: {
    marginVertical: 6,
    color: '#333',
  },
  blockquote: {
    borderLeftWidth: 4,
    borderLeftColor: '#43AA8B',
    paddingLeft: 10,
    marginVertical: 8,
    fontStyle: 'italic',
    color: '#666',
  },
  bullet_list: {
    marginVertical: 8,
  },
  ordered_list: {
    marginVertical: 8,
  },
  list_item: {
    flexDirection: 'row',
    paddingVertical: 4,
    color: '#333',
  },
  bullet_list_icon: {
    color: '#43AA8B',
    marginRight: 8,
    fontSize: 16,
  },
  code_inline: {
    backgroundColor: '#f5f5f5',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    fontFamily: 'Courier New',
    fontSize: 12,
    color: '#c41a3b',
  },
  code_block: {
    backgroundColor: '#f5f5f5',
    paddingHorizontal: 10,
    paddingVertical: 8,
    borderRadius: 6,
    marginVertical: 8,
    fontFamily: 'Courier New',
    fontSize: 12,
    color: '#333',
  },
  hr: {
    height: 1,
    backgroundColor: '#ddd',
    marginVertical: 12,
  },
  link: {
    color: '#43AA8B',
    textDecorationLine: 'underline',
  },
  table: {
    marginVertical: 8,
  },
  table_row: {
    flexDirection: 'row',
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
    paddingVertical: 6,
  },
  table_cell: {
    flex: 1,
    paddingHorizontal: 8,
    color: '#333',
  },
};

export default function ChatBotScreen({ route }) {
  const userData = route.params?.userData || {};
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const flatListRef = useRef(null);

  // ==================== LOAD CHAT HISTORY ====================
  useEffect(() => {
    loadChatHistory();
  }, []);

  const loadChatHistory = async () => {
    try {
      const stored = await AsyncStorage.getItem(CHAT_HISTORY_KEY);
      if (stored) {
        const history = JSON.parse(stored);
        // Convert timestamp strings back to Date objects
        const messagesWithDates = history.map(msg => ({
          ...msg,
          timestamp: msg.timestamp ? new Date(msg.timestamp) : new Date()
        }));
        setMessages(messagesWithDates);
        // FIX: Cuộn không animation và giảm timeout xuống mức tối thiểu (100ms) sau khi tải lịch sử
        setTimeout(() => {
          flatListRef.current?.scrollToEnd({ animated: false });
        }, 100); 
      }
    } catch (error) {
      console.error("Error loading chat history:", error);
    } finally {
      setLoadingHistory(false);
    }
  };

  // ==================== SAVE CHAT HISTORY ====================
  const saveChatHistory = async (updatedMessages) => {
    try {
      const toSave = updatedMessages.slice(-MAX_CHAT_HISTORY);
      await AsyncStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(toSave));
    } catch (error) {
      console.error("Error saving chat history:", error);
    }
  };

  // ==================== SEND MESSAGE ====================
  const handleSendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage = {
      id: Date.now().toString(),
      type: "user",
      text: inputText.trim(),
      timestamp: new Date(),
    };

    // 1. Thêm tin nhắn người dùng
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    saveChatHistory(newMessages);
    setInputText("");

    // 2. Cuộl xuống ngay lập tức sau khi gửi tin nhắn
    flatListRef.current?.scrollToEnd({ animated: true }); 

    setIsLoading(true);
    try {
      // Gọi API với user data
      const response = await sendChatMessage({
        message: userMessage.text,
        userData: {
          email: userData.email || "guest",
          name: userData.name || "User",
          health_info: {
            age: userData.age,
            gender: userData.gender,
            weight: userData.weight,
            height: userData.height,
            target: userData.target,
            allergies: userData.allergies || [],
            diseases: userData.diseases || [],
          },
        },
      });

      const botMessage = {
        id: (Date.now() + 1).toString(),
        type: "bot",
        text: response.reply || "Không thể xử lý yêu cầu của bạn. Vui lòng thử lại.",
        timestamp: new Date(),
        suggestions: response.suggestions || [],
      };

      // 3. Thêm tin nhắn bot
      const updatedMessages = [...newMessages, botMessage];
      setMessages(updatedMessages);
      saveChatHistory(updatedMessages);

      // 4. Cuộl xuống lần cuối
      flatListRef.current?.scrollToEnd({ animated: true });
      
    } catch (error) {
      console.error("Error sending message:", error);
      
      let errorMessageText = "❌ Lỗi kết nối. Vui lòng kiểm tra kết nối internet và thử lại.";

      if (error.message && (error.message.includes('timeout') || error.message.includes('network request failed'))) {
        errorMessageText = "⚠️ Yêu cầu quá lâu (Timeout). Vui lòng thử lại câu hỏi ngắn gọn hơn hoặc đợi server xử lý.";
      }

      const errorMessage = {
        id: (Date.now() + 1).toString(),
        type: "bot",
        text: errorMessageText,
        timestamp: new Date(),
      };

      // Thêm tin nhắn lỗi
      const updatedMessages = [...newMessages, errorMessage];
      setMessages(updatedMessages);
      saveChatHistory(updatedMessages);
      
      // Cuộl xuống sau khi thêm tin nhắn lỗi
      flatListRef.current?.scrollToEnd({ animated: true }); 

    } finally {
      setIsLoading(false);
    }
  };

  // ==================== CLEAR HISTORY ====================
  const handleClearHistory = () => {
    Alert.alert(
      "Xóa lịch sử",
      "Bạn có chắc muốn xóa toàn bộ lịch sử trò chuyện?",
      [
        { text: "Hủy", onPress: () => {} },
        {
          text: "Xóa",
          onPress: async () => {
            try {
              await AsyncStorage.removeItem(CHAT_HISTORY_KEY);
              setMessages([]);
            } catch (error) {
              console.error("Error clearing history:", error);
            }
          },
          style: "destructive",
        },
      ]
    );
  };

  // ==================== RENDER MESSAGE ====================
  const renderMessage = ({ item }) => {
    // Validation
    if (!item || typeof item.text !== 'string') {
      return null; // Bỏ qua tin nhắn không hợp lệ
    }

    const isUserMessage = item.type === "user";
    
    return (
      <View>
        <View style={[styles.messageContainer, isUserMessage ? styles.userMessageContainer : styles.botMessageContainer]}>
          {!isUserMessage && (<View style={styles.botIconContainer}><MaterialCommunityIcons name="robot-happy" size={24} color="#43AA8B" /></View>)}

          <View style={[styles.messageBubble, isUserMessage ? styles.userBubble : styles.botBubble]}>
            {isUserMessage ? (
              // Tin nhắn người dùng: hiển thị text bình thường
              <Text style={[styles.messageText, styles.userText]}>{item.text}</Text>
            ) : (
              // Tin nhắn bot: render markdown
              <MarkdownDisplay
                style={{
                  ...markdownStyles,
                  body: { ...markdownStyles.body, color: '#333' },
                  strong: { ...markdownStyles.strong, color: '#333' },
                  em: { ...markdownStyles.em, color: '#555' },
                }}
              >
                {item.text}
              </MarkdownDisplay>
            )}
            <Text style={[styles.timestamp, isUserMessage ? styles.userTimestamp : styles.botTimestamp]}>
              {item.timestamp && item.timestamp instanceof Date
                ? item.timestamp.toLocaleTimeString("vi-VN", {
                    hour: "2-digit",
                    minute: "2-digit",
                  })
                : item.timestamp
                ? new Date(item.timestamp).toLocaleTimeString("vi-VN", {
                    hour: "2-digit",
                    minute: "2-digit",
                  })
                : ""}
            </Text>
          </View>

          {isUserMessage && (<View style={styles.userIconContainer}><Ionicons name="person-circle" size={24} color="#43AA8B" /></View>)}
        </View>

        {/* Render suggestions if available - Below the message */}
        {!isUserMessage && item.suggestions && item.suggestions.length > 0 && (
          <View style={styles.suggestionsWrapperContainer}>
            {item.suggestions.map((suggestion, index) => (
              <TouchableOpacity
                key={index}
                style={styles.suggestionButton}
                onPress={() => { setInputText(suggestion); }}
              >
                <Ionicons name="arrow-forward" size={14} color="#43AA8B" style={styles.suggestionIcon} />
                <Text style={styles.suggestionText}>{suggestion}</Text>
              </TouchableOpacity>
            ))}
          </View>
        )}
      </View>
    );
  };

  // ==================== RENDER EMPTY STATE ====================
  const renderEmptyState = () => (
    <View style={styles.emptyContainer}>
      <MaterialCommunityIcons
        name="robot-happy-outline"
        size={80}
        color="#ccc"
      />
      <Text style={styles.emptyTitle}>Xin chào! 👋</Text>
      <Text style={styles.emptySubtitle}>
        Tôi là trợ lý ẩm thực của bạn. Hãy hỏi tôi bất cứ điều gì về:
      </Text>
      <View style={styles.emptyList}>
        <Text style={styles.emptyListItem}>🍎 Gợi ý thực phẩm phù hợp với bạn</Text>
        <Text style={styles.emptyListItem}>📊 Dinh dưỡng của các món ăn</Text>
        <Text style={styles.emptyListItem}>⚠️ Thực phẩm phù hợp với dị ứng/bệnh</Text>
        <Text style={styles.emptyListItem}>📋 Kế hoạch ăn uống cá nhân</Text>
      </View>
    </View>
  );

  if (loadingHistory) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#43AA8B" />
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.safeContainer}>
      <KeyboardAvoidingView
        style={styles.container}
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        keyboardVerticalOffset={Platform.OS === "ios" ? 0 : 20}
      >
        {/* HEADER */}
        <View style={styles.header}>
          <View style={styles.headerLeft}>
            <MaterialCommunityIcons name="robot-happy" size={28} color="#fff" />
            <View style={styles.headerTextContainer}>
              <Text style={styles.headerTitle}>Trợ Lý Ẩm Thực</Text>
              <Text style={styles.headerSubtitle}>Sẵn sàng giúp bạn 24/7</Text>
            </View>
          </View>
          <TouchableOpacity
            style={styles.headerButton}
            onPress={handleClearHistory}
            activeOpacity={0.6}
            hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
          >
            <Ionicons name="trash-outline" size={22} color="#fff" />
          </TouchableOpacity>
        </View>

        {/* MESSAGES */}
        {messages.length === 0 ? (
          renderEmptyState()
        ) : (
          <FlatList
            ref={flatListRef}
            data={messages}
            renderItem={renderMessage}
            keyExtractor={(item) => item.id}
            contentContainerStyle={styles.messageListContent}
          />
        )}

        {/* LOADING INDICATOR */}
        {isLoading && (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="small" color="#43AA8B" />
            <Text style={styles.loadingText}>Đang xử lý...</Text>
          </View>
        )}

        {/* INPUT AREA */}
        <View style={styles.inputArea}>
          <View style={styles.inputContainer}>
            <TextInput
              style={styles.textInput}
              placeholder="Hỏi tôi về thực phẩm, dinh dưỡng..."
              placeholderTextColor="#999"
              value={inputText}
              onChangeText={setInputText}
              multiline
              maxLength={500}
              editable={!isLoading}
            />
            <TouchableOpacity
              style={[
                styles.sendButton,
                (!inputText.trim() || isLoading) && styles.sendButtonDisabled,
              ]}
              onPress={handleSendMessage}
              disabled={!inputText.trim() || isLoading}
            >
              <Ionicons
                name="send"
                size={20}
                color={!inputText.trim() || isLoading ? "#ccc" : "#fff"}
              />
            </TouchableOpacity>
          </View>
          <Text style={styles.charCount}>
            {inputText.length}/500
          </Text>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeContainer: {
    flex: 1,
    backgroundColor: "#f5f5f5",
  },
  container: {
    flex: 1,
    backgroundColor: "#f5f5f5",
  },

  // HEADER
  header: {
    backgroundColor: "#43AA8B",
    paddingHorizontal: 15,
    paddingVertical: 12,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  headerLeft: {
    flexDirection: "row",
    alignItems: "center",
  },
  headerTextContainer: {
    marginLeft: 12,
  },
  headerTitle: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "bold",
  },
  headerSubtitle: {
    color: "#ddd",
    fontSize: 12,
  },
  headerButton: {
    padding: 12,
    borderRadius: 50,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "rgba(255, 255, 255, 0.2)",
  },

  // EMPTY STATE
  emptyContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 20,
  },
  emptyTitle: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#333",
    marginTop: 20,
  },
  emptySubtitle: {
    fontSize: 14,
    color: "#666",
    marginTop: 10,
    textAlign: "center",
  },
  emptyList: {
    marginTop: 20,
    alignItems: "flex-start",
    paddingLeft: 20,
  },
  emptyListItem: {
    fontSize: 13,
    color: "#555",
    marginVertical: 6,
  },

  // MESSAGES
  messageListContent: {
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  messageContainer: {
    flexDirection: "row",
    marginVertical: 8,
    alignItems: "flex-end",
  },
  userMessageContainer: {
    justifyContent: "flex-end",
  },
  botMessageContainer: {
    justifyContent: "flex-start",
  },

  messageBubble: {
    maxWidth: "80%",
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 16,
    marginHorizontal: 6,
  },
  userBubble: {
    backgroundColor: "#43AA8B",
    borderBottomRightRadius: 4,
  },
  botBubble: {
    backgroundColor: "#fff",
    borderBottomLeftRadius: 4,
    borderWidth: 1,
    borderColor: "#e0e0e0",
  },

  messageText: {
    fontSize: 14,
    lineHeight: 20,
  },
  userText: {
    color: "#fff",
  },
  botText: {
    color: "#333",
  },

  timestamp: {
    fontSize: 11,
    marginTop: 4,
  },
  userTimestamp: {
    color: "rgba(255, 255, 255, 0.7)",
  },
  botTimestamp: {
    color: "#999",
  },

  botIconContainer: {
    marginRight: 4,
  },
  userIconContainer: {
    marginLeft: 4,
  },

  // SUGGESTIONS
  suggestionsWrapperContainer: {
    paddingHorizontal: 50,
    paddingVertical: 8,
    flexDirection: "column",
    gap: 6,
    marginBottom: 8,
  },
  suggestionButton: {
    backgroundColor: "#f0f0f0",
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: "#43AA8B",
    flexDirection: "row",
    alignItems: "center",
  },
  suggestionIcon: {
    marginRight: 8,
    marginTop: 2,
  },
  suggestionText: {
    fontSize: 13,
    color: "#333",
    fontWeight: "500",
    flex: 1,
  },

  // LOADING
  loadingContainer: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 12,
    paddingVertical: 12,
  },
  loadingText: {
    marginLeft: 12,
    color: "#43AA8B",
    fontWeight: "500",
  },

  // INPUT AREA
  inputArea: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    backgroundColor: "#f5f5f5",
    borderTopWidth: 1,
    borderTopColor: "#ddd",
    marginBottom: 2,
  },
  inputContainer: {
    flexDirection: "row",
    alignItems: "flex-end",
    backgroundColor: "#fff",
    borderRadius: 24,
    paddingHorizontal: 12,
    paddingVertical: 2,
    borderWidth: 1,
    borderColor: "#ddd",
  },
  textInput: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 8,
    fontSize: 14,
    maxHeight: 100,
  },
  sendButton: {
    backgroundColor: "#43AA8B",
    padding: 10,
    borderRadius: 50,
    justifyContent: "center",
    alignItems: "center",
    marginLeft: 4,
  },
  sendButtonDisabled: {
    backgroundColor: "#ddd",
  },

  charCount: {
    fontSize: 11,
    color: "#999",
    marginTop: 6,
    textAlign: "right",
  },
});
