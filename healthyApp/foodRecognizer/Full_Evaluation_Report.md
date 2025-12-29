# BÁO CÁO THỰC NGHIỆM CHI TIẾT (NGƯỠNG 0.7)

## 1. Thông số Hệ thống
- **Mô hình:** CLIP ViT-B/32 (Zero-shot)
- **Số lượng mẫu thử (N):** 828
- **Ngưỡng tương đồng (Threshold):** 0.7
- **Thiết bị xử lý:** CPU Intel Core i7-11800H

## 2. Tỷ lệ Chính xác Tổng hợp
| Chỉ số Accuracy | Tỷ lệ (%) | Ý nghĩa |
| :--- | :--- | :--- |
| **Exact Top-1** | 5.07% | Khớp chính xác hoàn toàn nhãn gốc |
| **Semantic Top-1** | 82.73% | Khớp ngữ nghĩa (Similarity >= 0.7) |
| **Top-3 Hit Rate** | 11.47% | Nhãn gốc nằm trong 3 dự đoán đầu |
| **Latency Avg** | 0.174s | Thời gian xử lý trung bình/ảnh |

## 3. Nhật ký kiểm thử chi tiết (Toàn bộ mẫu)
| STT | Ground Truth | Top-1 (1K) | Top-3 Predictions (3K) | CosSim | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Dumplings | Packaged dumplings | 'Packaged dumplings', 'Assorted Dumplings', 'Meat Dumplings' | 0.903 | SEMANTIC |
| 2 | Boiled Peanuts | Boiled Peanuts | 'Boiled Peanuts', 'Spiced Peanuts', 'Lan Hua Dou (Peanut Snack)' | 1.000 | EXACT |
| 3 | Ice Cream Sundae | Sweet Taro Dessert | 'Sweet Taro Dessert', 'Decorative Ice Cream Dessert', 'Ube Ice Cream' | 0.737 | SEMANTIC |
| 4 | Gyudon (Beef Bowl) | Pork Rice Bowl with Egg | 'Pork Rice Bowl with Egg', 'Rice Bowl with Pork and Egg', 'Gyudon with Raw Egg' | 0.754 | SEMANTIC |
| 5 | Mixed Seafood and Meat Platter | Mixed Chinese Seafood and Meat Dishes | 'Mixed Chinese Seafood and Meat Dishes', 'Mixed Seafood and Pork Dishes', 'Mixed Seafood and Meat Dishes' | 0.861 | SEMANTIC |
| 6 | Spicy Crab | Crab with Stir-Fried Vegetables | 'Crab with Stir-Fried Vegetables', 'Stir-fried Crab', 'Stir-fried Crab with Vegetables' | 0.767 | SEMANTIC |
| 7 | Fried Eggs | Egg Soufflé | 'Egg Soufflé', 'Cheese Soufflé', 'Cheesy Baked Egg' | 0.727 | SEMANTIC |
| 8 | Spaghetti and Meatballs | Sausage and Meatballs with Noodles | 'Sausage and Meatballs with Noodles', 'Meatballs with Noodles', 'Noodles with Meatballs' | 0.897 | SEMANTIC |
| 9 | Cheese and Mushroom Pizza | Cheese and Mushroom Pizza | 'Cheese and Mushroom Pizza', 'Mango Pizza', 'Cheese and Pineapple Pizza' | 1.000 | EXACT |
| 10 | Assorted Pastries | Mini Pastries | 'Mini Pastries', 'Mini pastries', 'Assorted Pastries' | 0.910 | SEMANTIC |
| 11 | Pizza with Fries and Chicken Wings | Themed Pizza and Burger Meal | 'Themed Pizza and Burger Meal', 'Flatbreads with assorted sides', 'Pizza, Burger, Fries, Salad' | 0.716 | SEMANTIC |
| 12 | Fried Chicken | KFC Fried Chicken Meal | 'KFC Fried Chicken Meal', 'Fried Snack with Meat Floss', 'Fried Chicken and Stir-fried Meat' | 0.903 | SEMANTIC |
| 13 | Mixed Asian Noodle Dish | Sukiyaki Beef | 'Sukiyaki Beef', 'Noodle Bowl with Meat and Egg', 'Jiaomijiao Pickled Pepper Beef Noodle' | 0.747 | SEMANTIC |
| 14 | Strawberry French Toast | Stuffed Pastries with Tomatoes | 'Stuffed Pastries with Tomatoes', 'Sweet Bread with Fruit Jellies', 'Bread with Honey and Tomatoes' | 0.644 | MISS |
| 15 | Mixed Meat Rice | Vietnamese Meat Dish with Rice Crackers | 'Vietnamese Meat Dish with Rice Crackers', 'Sweet and Sour Pork with Rice', 'Rice with Sausage and Green Beans' | 0.783 | SEMANTIC |
| 16 | Grapes | Kyoho Grapes | 'Kyoho Grapes', 'Mixed Grapes', 'Seedless Grapes' | 0.869 | SEMANTIC |
| 17 | Grilled Chicken with Vegetables | Chicken and Vegetable Box | 'Chicken and Vegetable Box', 'Mixed Chicken Bento', 'Mixed Meal Tray' | 0.759 | SEMANTIC |
| 18 | Vegetable Noodles | Stir-fried Noodles with Beef and Spinach | 'Stir-fried Noodles with Beef and Spinach', 'Stir-fried flat noodles', 'Spicy Stir-Fried Noodles' | 0.743 | SEMANTIC |
| 19 | Noodle and Wrap Combo | Mixed Seafood and Chicken Lunchbox | 'Mixed Seafood and Chicken Lunchbox', 'Char Kway Teow', 'Steamed Bun and Chicken Feet' | 0.692 | MISS |
| 20 | Cake | Decorative Birthday Cake | 'Decorative Birthday Cake', 'Celebration Cake', 'Decorative Cake' | 0.903 | SEMANTIC |
| 21 | Pastry | McDonald's Pie and Soft Drink | 'McDonald's Pie and Soft Drink', 'Flan with a drink', 'Foam Topped Beverage' | 0.742 | SEMANTIC |
| 22 | Pho | Vietnamese Chicken Soup | 'Vietnamese Chicken Soup', 'Vietnamese Noodle Soup', 'Bún Bò Huế' | 0.800 | SEMANTIC |
| 23 | Fried Chicken with Rice | Nasi Kuning with Chicken | 'Nasi Kuning with Chicken', 'Nasi Kuning with Fried Chicken', 'Grilled Pork Ribs with Rice' | 0.762 | SEMANTIC |
| 24 | Stir-fried Chicken with Vegetables | Spicy Meat Soup and Stir-Fried Dish | 'Spicy Meat Soup and Stir-Fried Dish', 'Spicy Sichuan Cuisine', 'Stir-fried beef with vegetables and soup' | 0.806 | SEMANTIC |
| 25 | Fried seafood platter | Dim Sum and Fried Chicken | 'Dim Sum and Fried Chicken', 'Mixed Fried Chicken Dishes', 'Mixed platter with fried chicken and sides' | 0.704 | SEMANTIC |
| 26 | Plums | Fresh Plums | 'Fresh Plums', 'Red Plums', 'Plums' | 0.950 | SEMANTIC |
| 27 | Canned Vegetables | Tsingtao Beer | 'Tsingtao Beer', 'Mung Bean Drink', 'Canned Green Peas' | 0.677 | MISS |
| 28 | Korean Hot Pot | Spicy Hot Pot with Noodles and Vegetables | 'Spicy Hot Pot with Noodles and Vegetables', 'Hot Pot with Noodles', 'Korean Mixed Rice (Bibimbap)' | 0.810 | SEMANTIC |
| 29 | Crackers | Packaged Rice Cake | 'Packaged Rice Cake', 'Red Bean Cake', 'Lan Hua Dou (Peanut Snack)' | 0.785 | SEMANTIC |
| 30 | Stir-fried Beef with Vegetables | Stir-fried Liver with Green Peppers | 'Stir-fried Liver with Green Peppers', 'Stir-fried Beef with Green Onions', 'Stir-fried Pork Liver' | 0.824 | SEMANTIC |
| 31 | Mixed Salad | Protein Salad with Poached Egg | 'Protein Salad with Poached Egg', 'Salad with Poached Egg', 'Green Salad with Dressing' | 0.757 | SEMANTIC |
| 32 | Shrimp | Seafood (e.g., Lobster) | 'Seafood (e.g., Lobster)', 'Shrimp Cocktail', 'Creamy Shrimp' | 0.842 | SEMANTIC |
| 33 | Mixed Asian Cuisine | Tonkatsu set | 'Tonkatsu set', 'Noodle and Tempura Set', 'Japanese Izakaya Meal' | 0.683 | MISS |
| 34 | Mandarin Orange | Grapefruit segments | 'Grapefruit segments', 'Orange segments', 'Orange Segments' | 0.752 | SEMANTIC |
| 35 | Steamed Chicken | Chicken Fat or Similar Dish | 'Chicken Fat or Similar Dish', 'Xue Yan (Snow Fungus)', 'Steamed Chicken with Cabbage' | 0.832 | SEMANTIC |
| 36 | Apple | Fruit-filled Bagel | 'Fruit-filled Bagel', 'Mango Custard Bun', 'Yellow fruit (e.g., apricot)' | 0.740 | SEMANTIC |
| 37 | Stir-fried Bok Choy | Stir-fried Green Beans with Seeds | 'Stir-fried Green Beans with Seeds', 'Stir-fried Green Vegetables with Nuts', 'Stir-fried Green Beans with Black Fungus' | 0.645 | MISS |
| 38 | Borscht | Radish Soup | 'Radish Soup', 'Borscht', 'Borscht with Bread and Cheese' | 0.799 | SEMANTIC |
| 39 | Grilled Chicken | Fried Skewered Chicken | 'Fried Skewered Chicken', 'Skewered Meat and Fried Snacks', 'Fried Meat Skewer' | 0.847 | SEMANTIC |
| 40 | Braised Pork | Stewed Chicken Feet | 'Stewed Chicken Feet', 'Stir-fried Chicken with Black Fungus', 'Braised Chicken with Century Eggs' | 0.774 | SEMANTIC |
| 41 | Seafood and Noodle Platter | Steamed Crabs with Side Dishes | 'Steamed Crabs with Side Dishes', 'Steamed Crab with Side Dishes', 'Chili Crab' | 0.732 | SEMANTIC |
| 42 | Vegetable Stew with Noodles | Spicy Soup with Fried Snacks | 'Spicy Soup with Fried Snacks', 'Herbal Fish Soup', 'Vegetable Hot Pot' | 0.760 | SEMANTIC |
| 43 | Steamed Fish with Vegetables | Seafood Paella | 'Seafood Paella', 'Steamed Fish with Spicy Toppings', 'Seafood Rice and Assorted Dishes' | 0.667 | MISS |
| 44 | Steamed Fish | Steamed Fish with Vegetables and Rice | 'Steamed Fish with Vegetables and Rice', 'Steamed Fish with Rice and Vegetables', 'Steamed Fish with Vegetables' | 0.835 | SEMANTIC |
| 45 | Roasted Purple Potatoes | Purple Sweet Potato | 'Purple Sweet Potato', 'Purple Sweet Potato Snack', 'Steamed Purple Sweet Potato' | 0.855 | SEMANTIC |
| 46 | Spicy Shrimp | Spicy Stir-Fried Frog Legs | 'Spicy Stir-Fried Frog Legs', 'Stir-fried Frog Legs', 'Spicy Stir-Fried Shrimp' | 0.777 | SEMANTIC |
| 47 | Stuffed Flatbread | Flatbread with sauce | 'Flatbread with sauce', 'Flatbread with Sauce', 'Naan' | 0.903 | SEMANTIC |
| 48 | Steamed Dumplings | Momos | 'Momos', 'Rice Dumpling', 'Creamy Dumplings' | 0.789 | SEMANTIC |
| 49 | Rice with meat and vegetables | Mixed Rice Plate with Beef | 'Mixed Rice Plate with Beef', 'Mixed Plate with Rice and Meat', 'Rice with meat and pickled vegetables' | 0.879 | SEMANTIC |
| 50 | Stir-fried Cucumbers with Pork | Stir-fried Pork with Cucumbers | 'Stir-fried Pork with Cucumbers', 'Stir-fried cucumber with meat', 'Stir-fried Cucumber with Meat' | 0.949 | SEMANTIC |
| 51 | Oysters | Oysters with dipping sauces | 'Oysters with dipping sauces', 'Steamed Oysters with Glass Noodles', 'Oyster with sauce' | 0.847 | SEMANTIC |
| 52 | Creamy Chicken | Sweet Dumplings in Coconut Milk | 'Sweet Dumplings in Coconut Milk', 'Coconut Chicken Curry', 'Chicken with herbs and soup' | 0.764 | SEMANTIC |
| 53 | Seafood Noodle Dish | Spicy Noodles with Fried Dumplings | 'Spicy Noodles with Fried Dumplings', 'Shrimp Noodles with Fried Fish', 'Spicy Noodles with Vegetables and Seafood' | 0.821 | SEMANTIC |
| 54 | Steamed Seafood with Noodles | Steamed Scallops with Vermicelli | 'Steamed Scallops with Vermicelli', 'Stuffed Clams with Noodles', 'Steamed Scallops with Shrimp' | 0.822 | SEMANTIC |
| 55 | Grilled Meat | Steamed Meat with Dipping Sauce | 'Steamed Meat with Dipping Sauce', 'Sliced Meat with Dipping Sauce', 'Steamed Meat Platter' | 0.812 | SEMANTIC |
| 56 | Fried Chicken Wings | Stir-fried Bamboo Shoots with Meat | 'Stir-fried Bamboo Shoots with Meat', 'Chicken Feet Dish', 'Stir-fried Chicken Feet' | 0.660 | MISS |
| 57 | Spicy Ribs | Spicy Fried Pork Ribs | 'Spicy Fried Pork Ribs', 'Spicy Fried Ribs', 'Stir-Fried Chicken Wings' | 0.920 | SEMANTIC |
| 58 | Shrimp Stir-Fry | Spicy Chicken Feet Salad | 'Spicy Chicken Feet Salad', 'Spicy Stir-Fried Seafood and Vegetables', 'Spicy Stir-Fried Seafood and Soup' | 0.708 | SEMANTIC |
| 59 | Noodle Soup with Meat and Egg | Gyudon with Raw Egg | 'Gyudon with Raw Egg', 'Steamed Meat and Egg Dish', 'Steamed Fish with Eggs' | 0.747 | SEMANTIC |
| 60 | Glazed Ribs | Sweet Soy Glazed Tofu | 'Sweet Soy Glazed Tofu', 'Sweet and Sour Pork Ribs', 'Stir-fried Pork Belly' | 0.712 | SEMANTIC |
| 61 | Soymilk | Packaged Rice Ball | 'Packaged Rice Ball', 'Packaged Rice Cake', 'Packaged Rice Noodles' | 0.763 | SEMANTIC |
| 62 | Noodle Soup | Noodle Soup with Chicken and Pumpkin | 'Noodle Soup with Chicken and Pumpkin', 'Korean Soup (likely Galbitang)', 'Meat Stew with Rice Noodles' | 0.857 | SEMANTIC |
| 63 | Dumplings | Frozen Musang King Durian Pulp | 'Frozen Musang King Durian Pulp', 'Steamed Yellow Squash', 'Jackfruit' | 0.597 | MISS |
| 64 | Roast Pork | Glazed Pork Belly | 'Glazed Pork Belly', 'Grilled Pork Belly', 'Char Siu (Chinese BBQ Pork)' | 0.841 | SEMANTIC |
| 65 | Vegetable Stir-Fried Noodles | Fried Noodles with Vegetables | 'Fried Noodles with Vegetables', 'Fried Noodles with Crispy Topping', 'Vegetable Fried Noodles' | 0.927 | SEMANTIC |
| 66 | Banana | Banana | 'Banana', 'Sliced Bananas', 'Banana-shaped cake' | 1.000 | EXACT |
| 67 | Milk | Matcha-flavored snack | 'Matcha-flavored snack', 'Seaweed-flavored snack', 'Soy Milk Flavor Snack' | 0.765 | SEMANTIC |
| 68 | Mixed Vegetable Rice Meal | Korean meal with rice and side dishes | 'Korean meal with rice and side dishes', 'Korean Meal Set', 'Korean multi-course meal' | 0.755 | SEMANTIC |
| 69 | Bananas | Sliced Bananas | 'Sliced Bananas', 'Banana', 'Snacks with Bananas' | 0.901 | SEMANTIC |
| 70 | Noodle Bowl | Jajangmyeon or similar noodle dish | 'Jajangmyeon or similar noodle dish', 'Spicy Pork Bone Noodles', 'Sichuan Spicy Noodles' | 0.760 | SEMANTIC |
| 71 | Stir-fried leafy greens | Stir-fried Water Spinach | 'Stir-fried Water Spinach', 'Stir-fried greens with sweet soup', 'Vegetable Soup with Stir-Fried Greens' | 0.877 | SEMANTIC |
| 72 | Grilled Shrimp | Grilled Shrimp with Salad | 'Grilled Shrimp with Salad', 'Marinated Shrimp', 'Boiled Shrimp' | 0.869 | SEMANTIC |
| 73 | Seaweed Soup | Seaweed Soup with Rice | 'Seaweed Soup with Rice', 'Vegetable Seaweed Soup', 'Seaweed Soup' | 0.924 | SEMANTIC |
| 74 | Hot Pot | Herbal Meat Soup | 'Herbal Meat Soup', 'Vegetable Hot Pot', 'Meat and Vegetable Hot Pot' | 0.813 | SEMANTIC |
| 75 | Braised Ribs | Stir-Fried Chicken Wings | 'Stir-Fried Chicken Wings', 'Sweet and Sour Pork Ribs', 'Sichuan Chicken' | 0.749 | SEMANTIC |
| 76 | Dumpling Soup | Sweet Dumplings in Ginger Syrup | 'Sweet Dumplings in Ginger Syrup', 'Noodle Soup with Dumplings', 'Dumplings with Noodles' | 0.851 | SEMANTIC |
| 77 | Cooked White Rice | Steamed Rice or Grain Dish | 'Steamed Rice or Grain Dish', 'Rice with Umeboshi', 'Rice with Soft Drink' | 0.892 | SEMANTIC |
| 78 | Buckwheat with Meat and Vegetables | Buckwheat with Meat and Vegetables | 'Buckwheat with Meat and Vegetables', 'Meat with Buckwheat', 'Buckwheat with Carrot Salad' | 1.000 | EXACT |
| 79 | Sashimi Platter | Sashimi Platter with Rice | 'Sashimi Platter with Rice', 'Salmon Sashimi with Roe', 'Sashimi with Fish' | 0.940 | SEMANTIC |
| 80 | Spicy Noodle Bowl | Noodle Soup with Beans | 'Noodle Soup with Beans', 'Spicy Noodles with Peanuts', 'Noodle dish with peanuts' | 0.846 | SEMANTIC |
| 81 | Baked goods with dessert | Fried Bread with Drink | 'Fried Bread with Drink', 'Fried Bread and Drink', 'Fried pastry with beverage' | 0.812 | SEMANTIC |
| 82 | Vegetable Wrap | Flatbread with vegetable and protein dish | 'Flatbread with vegetable and protein dish', 'Chickpea Wrap', 'Savory Pancake Wrap' | 0.813 | SEMANTIC |
| 83 | Sweet Fruit Soup | Lotus Seed Sweet Soup | 'Lotus Seed Sweet Soup', 'Mung Bean Soup with Glutinous Rice Balls', 'Tangyuan' | 0.799 | SEMANTIC |
| 84 | Pasta with Tomato Sauce | Penne Pasta with Tomato Sauce | 'Penne Pasta with Tomato Sauce', 'Pasta with Chicken and Tomato Sauce', 'Pasta with Tomato Sauce and Vegetables' | 0.940 | SEMANTIC |
| 85 | Stir-fried Beef with Vegetables | Stir-fried Snap Peas with Pork | 'Stir-fried Snap Peas with Pork', 'Stir-fried greens with meat and soup', 'Stir-fried Snow Peas with Pork' | 0.727 | SEMANTIC |
| 86 | Stuffed Green Peppers | Cucumber Sticks | 'Cucumber Sticks', 'Sugar Cane Candy', 'Cucumber Sticks with Sauce' | 0.614 | MISS |
| 87 | Cheese Pizza | Cheese Pizza with Pineapple | 'Cheese Pizza with Pineapple', 'Cheese and Pineapple Pizza', 'Cheese Pizza' | 0.889 | SEMANTIC |
| 88 | Zongzi | Zongzi (Rice Dumpling) | 'Zongzi (Rice Dumpling)', 'Zongzi', 'Steamed Fish in Banana Leaves' | 0.948 | SEMANTIC |
| 89 | Baked Oysters | Boiled Egg with Sweet Potato and Spinach | 'Boiled Egg with Sweet Potato and Spinach', 'Decorative Fish and Egg Platter', 'Quail Eggs with Accompaniments' | 0.554 | MISS |
| 90 | Chicken Stir-Fry | Vegetable Curry with Chicken | 'Vegetable Curry with Chicken', 'Chicken Curry with Potatoes', 'Stir-fried Bamboo Shoots with Pork' | 0.847 | SEMANTIC |
| 91 | Grapes | Seedless Grapes | 'Seedless Grapes', 'Kyoho Grapes', 'Grape Snack' | 0.885 | SEMANTIC |
| 92 | Sweet Cake | Packaged Rice Ball | 'Packaged Rice Ball', 'Packaged Rice Cake', 'Mango Custard Bun' | 0.775 | SEMANTIC |
| 93 | Fried Chicken Stir-Fry | Spicy Stir-Fried Chicken with Peppers | 'Spicy Stir-Fried Chicken with Peppers', 'Stir-fried Chicken with Bell Peppers', 'Spicy Chicken with Peppers' | 0.883 | SEMANTIC |
| 94 | Lobster with Spices | Crab with topping | 'Crab with topping', 'Fried Crab with Rice', 'Black Pepper Crab' | 0.837 | SEMANTIC |
| 95 | Mixed Grill | Korean Meal Set | 'Korean Meal Set', 'Mixed Meal Tray', 'Korean multi-course meal' | 0.706 | SEMANTIC |
| 96 | Vegetable Soup | Seaweed Soup with Rice | 'Seaweed Soup with Rice', 'Vegetable Seaweed Soup', 'Egg Seaweed Soup' | 0.803 | SEMANTIC |
| 97 | Cake | Cute Animal Pastries | 'Cute Animal Pastries', 'Pastry with Cream and Caramel', 'Almond Cake' | 0.776 | SEMANTIC |
| 98 | Fruit-flavored snack | Packaged Rice Ball | 'Packaged Rice Ball', 'Packaged Rice Cake', 'Strawberry Milk Jelly' | 0.785 | SEMANTIC |
| 99 | Fried Chicken Wings | Fried Sausage on a Stick | 'Fried Sausage on a Stick', 'Fried Skewered Chicken', 'Fried Snack on a Stick' | 0.768 | SEMANTIC |
| 100 | Grilled Meat Platter | Korean BBQ with side dishes | 'Korean BBQ with side dishes', 'Korean BBQ Pork', 'Korean BBQ' | 0.751 | SEMANTIC |
| 101 | Breakfast Platter | Salmon and Shrimp Breakfast Plate | 'Salmon and Shrimp Breakfast Plate', 'Mixed Breakfast Platter', 'Croissant with Fruits and Coffee' | 0.784 | SEMANTIC |
| 102 | Baked Oysters | Baked Oysters and Scallops | 'Baked Oysters and Scallops', 'Steamed Oysters with Glass Noodles', 'Spicy Oysters' | 0.928 | SEMANTIC |
| 103 | Vegetable Stew | Meat Stew with Dried Fruits | 'Meat Stew with Dried Fruits', 'Meat Stew with Peppers and Beans', 'Beef Stew with Pineapple' | 0.829 | SEMANTIC |
| 104 | Fruit Drink | Fermented Milk Drink | 'Fermented Milk Drink', 'Unknown product', 'Lactic Acid Bacteria Drink' | 0.871 | SEMANTIC |
| 105 | Chicken Soup | Samgyetang (Korean Ginseng Chicken Soup) | 'Samgyetang (Korean Ginseng Chicken Soup)', 'Samgyetang (Ginseng Chicken Soup)', 'Korean Soup (likely Samgyetang)' | 0.667 | MISS |
| 106 | Sweet and Spicy Chicken Legs | BBQ Chicken Drumsticks | 'BBQ Chicken Drumsticks', 'Spicy Glazed Chicken', 'BBQ Chicken Wings' | 0.809 | SEMANTIC |
| 107 | Assorted Dishes with Rice | Skewered Hot Pot | 'Skewered Hot Pot', 'Dim Sum Platter', 'Dim Sum' | 0.715 | SEMANTIC |
| 108 | Fried rice with fritters | Rice Balls with Chicken | 'Rice Balls with Chicken', 'Fried Chicken with Curry and Rice', 'Rice with Fish Patties' | 0.808 | SEMANTIC |
| 109 | Sliced Roast Meat | Sliced Liver Dish | 'Sliced Liver Dish', 'Sliced Pork with Side Dishes', 'Meat and Nuts Platter' | 0.891 | SEMANTIC |
| 110 | Stir-fried Green Vegetables with Fish | Stir-fried Seaweed Salad | 'Stir-fried Seaweed Salad', 'Stir-fried Fiddlehead Ferns', 'Stir-fried Water Spinach' | 0.763 | SEMANTIC |
| 111 | Fried Chicken Ramen | Seafood Congee | 'Seafood Congee', 'Fish Congee', 'Shrimp Congee' | 0.699 | MISS |
| 112 | Rice with Black Tofu | Rice with Black Tofu | 'Rice with Black Tofu', 'Rice Porridge with Red Dates', 'Rice with Red Dates' | 1.000 | EXACT |
| 113 | Steamed Fish | Steamed Fish with Sauce | 'Steamed Fish with Sauce', 'Steamed Fish with Seafood', 'Steamed Fish in Soy Sauce' | 0.940 | SEMANTIC |
| 114 | Fried Chicken Sandwich | Fried Chicken Roll | 'Fried Chicken Roll', 'Chinese Rice Burger', 'Fried Chicken Sandwich with Fried Chicken' | 0.899 | SEMANTIC |
| 115 | Iced Beverage | Starbucks Frappuccino and Iced Drink | 'Starbucks Frappuccino and Iced Drink', 'Frappuccino', 'Cake and Frappuccino' | 0.791 | SEMANTIC |
| 116 | Dumplings | Mixed Plate with Steamed Dumplings | 'Mixed Plate with Steamed Dumplings', 'Steamed Dumplings with Yellow Soup', 'Savory Steamed Dumplings' | 0.872 | SEMANTIC |
| 117 | Watermelon | Cut Watermelon | 'Cut Watermelon', 'Watermelon and Snacks', 'Watermelon slices' | 0.919 | SEMANTIC |
| 118 | Spicy Clams | Spicy Clams with Vegetables | 'Spicy Clams with Vegetables', 'Stir-fried vegetables and clams', 'Steamed Shellfish with Vegetables' | 0.898 | SEMANTIC |
| 119 | General Tso's Chicken | Korean Fried Chicken with Rice Cakes | 'Korean Fried Chicken with Rice Cakes', 'Korean Fried Chicken', 'Stir-Fried Chicken Wings' | 0.791 | SEMANTIC |
| 120 | Mixed Seafood and Skewers | Mixed Chinese Seafood and Meat Dishes | 'Mixed Chinese Seafood and Meat Dishes', 'Crawfish and assorted grilled dishes', 'Crawfish and assorted dishes' | 0.866 | SEMANTIC |
| 121 | Shrimp Chow Mein | Stir-fried Noodles with Shrimp and Beef | 'Stir-fried Noodles with Shrimp and Beef', 'Stir-fried Noodles with Shrimp and Pork', 'Stir-fried Glass Noodles with Shrimp' | 0.863 | SEMANTIC |
| 122 | Fruit Pie | Meat-filled pastry | 'Meat-filled pastry', 'Fruit-filled pastry', 'Savory Galette' | 0.836 | SEMANTIC |
| 123 | Green Dessert with Dumplings | Matcha dessert with red bean | 'Matcha dessert with red bean', 'Green Pea Pudding', 'Green Dessert with Dumplings' | 0.778 | SEMANTIC |
| 124 | Garlic Eggplant | Stuffed Eggplant with Couscous | 'Stuffed Eggplant with Couscous', 'Fried Eggplant with Minced Meat', 'Stuffed Eggplant with Rice' | 0.746 | SEMANTIC |
| 125 | Mixed Vegetable Rice Plate | Traditional Holiday Plate | 'Traditional Holiday Plate', 'Rice with Fish and Accompaniments', 'Rice Dish with Accompaniments' | 0.778 | SEMANTIC |
| 126 | Meat Stew | Stir-fried bamboo shoots with pork | 'Stir-fried bamboo shoots with pork', 'Stir-fried Bamboo Shoots with Pork', 'Stir-fried meat with bamboo shoots' | 0.700 | SEMANTIC |
| 127 | Fried Fish with Sauce | Braised Sea Cucumber | 'Braised Sea Cucumber', 'Sea Cucumber Dish', 'Marinated Sea Cucumber' | 0.737 | SEMANTIC |
| 128 | Steamed Lettuce | Stir-fried Water Spinach | 'Stir-fried Water Spinach', 'Steamed Fish with Greens', 'Stir-fried Greens with Sauce' | 0.641 | MISS |
| 129 | Steamed Chicken | Chicken Fat or Similar Dish | 'Chicken Fat or Similar Dish', 'Boiled Chicken Parts', 'Lemon Chicken' | 0.832 | SEMANTIC |
| 130 | Braised Meat Dish | Liver with sauce | 'Liver with sauce', 'Cooked Organ Meat in Sauce', 'Braised Meat with Sauce' | 0.835 | SEMANTIC |
| 131 | Noodle Salad | Stir-fried Cabbage with Meat and Spicy Cucumber Salad | 'Stir-fried Cabbage with Meat and Spicy Cucumber Salad', 'Stir-fried Cabbage and Bean Sprouts', 'Steamed Rice with Fried Shallots' | 0.615 | MISS |
| 132 | Pasta with Broccoli and Sausage | Pasta with Sausage and Broccoli | 'Pasta with Sausage and Broccoli', 'Pasta with Broccoli and Sausage', 'Shrimp Pasta with Broccoli' | 0.988 | SEMANTIC |
| 133 | Steamed Seafood | Tai Ma Hua (Sesame Snack) | 'Tai Ma Hua (Sesame Snack)', 'Fried Meat with Sesame', 'Steamed Fish and Sliced Meat' | 0.650 | MISS |
| 134 | Noodle with Sausages | Scrambled Eggs with Hot Dogs | 'Scrambled Eggs with Hot Dogs', 'Sausages with mashed potatoes', 'Hot Dogs with Toppings' | 0.726 | SEMANTIC |
| 135 | Fried Chicken and Fries | Fried food with fries | 'Fried food with fries', 'Fried Shrimp and Fries', 'Mixed Fried Platter' | 0.888 | SEMANTIC |
| 136 | Peaches | Canned Yellow Peaches | 'Canned Yellow Peaches', 'Cooked Pears', 'Yellow Plums' | 0.775 | SEMANTIC |
| 137 | Dumplings | Vietnamese steamed rice cakes | 'Vietnamese steamed rice cakes', 'Steamed Rice Balls', 'Steamed Dough Buns' | 0.735 | SEMANTIC |
| 138 | Banana | Banana | 'Banana', 'Sliced Bananas', 'Banana on a stick' | 1.000 | EXACT |
| 139 | Grilled Sausages with Sauce | Assorted Sausages and Side Dishes | 'Assorted Sausages and Side Dishes', 'Sausages with boiled egg and cucumber', 'Sausages with side dishes' | 0.796 | SEMANTIC |
| 140 | Spaghetti Bolognese | Baked Dish with Meat Sauce | 'Baked Dish with Meat Sauce', 'Spaghetti with Meat Sauce and Sausage', 'Spaghetti with Tomato Sauce and Mozzarella' | 0.775 | SEMANTIC |
| 141 | Packaged meal | Jiao Yan Su Biscuits | 'Jiao Yan Su Biscuits', 'Ginseng Candy', 'Packaged Egg Snack' | 0.702 | SEMANTIC |
| 142 | Fried Noodles | Chicken Feet Stir-Fry | 'Chicken Feet Stir-Fry', 'Stir-fried Green Beans with Black Fungus', 'Spicy Chicken Feet Salad' | 0.736 | SEMANTIC |
| 143 | Boiled Peanuts | Boiled Peanuts | 'Boiled Peanuts', 'Spiced Peanuts', 'Fried Edible Insects' | 1.000 | EXACT |
| 144 | Meat and Vegetable Soup | Pork Soup with Corn | 'Pork Soup with Corn', 'Pork and Corn Stew', 'Corn and Pork Stew' | 0.808 | SEMANTIC |
| 145 | Spicy Fried Pork | Sichuan Chicken | 'Sichuan Chicken', 'Spicy Stir-Fried Chicken', 'Stir-fried Chicken with Mushrooms' | 0.838 | SEMANTIC |
| 146 | Vegetable Noodles | Stir-fried Bean Sprouts and Peppers | 'Stir-fried Bean Sprouts and Peppers', 'Shredded Green Papaya Salad', 'Stir-fried Glass Noodles with Vegetables' | 0.768 | SEMANTIC |
| 147 | Braised Short Ribs | Sweet and Sour Pork Ribs | 'Sweet and Sour Pork Ribs', 'Stir-Fried Chicken Wings', 'Sweet Soy Chicken Wings' | 0.800 | SEMANTIC |
| 148 | Baked pastry with iced coffee | Mango Lassi or similar dessert | 'Mango Lassi or similar dessert', 'Tahu Telor', 'Fried Bread with Drink' | 0.679 | MISS |
| 149 | Stir-Fried Chicken with Vegetables | Stir-fried Chicken with Snow Peas | 'Stir-fried Chicken with Snow Peas', 'Stir-fried Chicken with Green Pepper', 'Stir-fried Chicken with Peas' | 0.795 | SEMANTIC |
| 150 | Raw Pork | Vacuum-sealed fish | 'Vacuum-sealed fish', 'Vacuum-sealed meat', 'Packaged Duck Legs' | 0.799 | SEMANTIC |
| 151 | Noodle Salad | Chicken Rice with Salad | 'Chicken Rice with Salad', 'Noodle Salad with Crispy Pancake', 'Beef Noodles with Salad and Fried Cakes' | 0.727 | SEMANTIC |
| 152 | Sashimi | Tuna Sashimi | 'Tuna Sashimi', 'Tuna Sushi', 'Canned Yellowfin Tuna' | 0.867 | SEMANTIC |
| 153 | Fruit Salad | Mango and Blueberry Salad | 'Mango and Blueberry Salad', 'Frozen Mixed Fruit', 'Mixed Snacks and Fruits' | 0.718 | SEMANTIC |
| 154 | Stir-fried Noodles | Noodle Bowl with Hot Dog | 'Noodle Bowl with Hot Dog', 'Instant Noodles with Sausage and Vegetables', 'Noodles with fried snacks' | 0.752 | SEMANTIC |
| 155 | Stir-fried Noodles | Spicy Shredded Chicken | 'Spicy Shredded Chicken', 'Stir-fried Cabbage with Pork', 'Rice with shredded meat' | 0.786 | SEMANTIC |
| 156 | Stir-fried Chicken with Greens | Stir-fried Chicken with Black Fungus | 'Stir-fried Chicken with Black Fungus', 'Fried Chicken and Stir-fried Meat', 'Stir-fried beef and chicken feet' | 0.807 | SEMANTIC |
| 157 | Mixed Asian Dish | Mixed Asian Dishes with Rice | 'Mixed Asian Dishes with Rice', 'Stir-fried greens with pork and side dishes', 'Mixed Meat Dishes with Rice' | 0.942 | SEMANTIC |
| 158 | Fish and Vegetable Soup | Vietnamese Hot Pot | 'Vietnamese Hot Pot', 'Bún riêu', 'Steamed Meat with Dipping Sauce' | 0.671 | MISS |
| 159 | Noodle Soup | Tsukemen (Dipping Noodles) | 'Tsukemen (Dipping Noodles)', 'Noodle dish with meat', 'Noodle Dish with Meat' | 0.670 | MISS |
| 160 | Beef Noodle Soup | Noodle Soup with Skewered Meat | 'Noodle Soup with Skewered Meat', 'Jajangmyeon or similar noodle dish', 'Spicy Beef Noodles with Skewers' | 0.865 | SEMANTIC |
| 161 | Noodle Soup with Fried Snacks | Vietnamese Spring Rolls and Noodle Soup | 'Vietnamese Spring Rolls and Noodle Soup', 'Vietnamese meal with rice and assorted dishes', 'Assorted Vietnamese Dishes' | 0.792 | SEMANTIC |
| 162 | Milkshake | Pepero | 'Pepero', 'Bubble Tea and Whipped Cream Drink', 'Ice Cream Bar' | 0.732 | SEMANTIC |
| 163 | Grilled Vegetable and Shrimp Medley | Seafood Boil with Dumplings | 'Seafood Boil with Dumplings', 'Spicy Seafood Boil', 'Grilled Shrimp and Vegetable Medley' | 0.667 | MISS |
| 164 | Seafood Platter | Fried Edible Insects | 'Fried Edible Insects', 'Assorted Edible Insects', 'Fried Insects with Vegetables' | 0.677 | MISS |
| 165 | Dried Fish Snack | Hi-Protein Jerky with Honey Flavour | 'Hi-Protein Jerky with Honey Flavour', 'Dried Edible Insects', 'Dried Beef Snack' | 0.709 | SEMANTIC |
| 166 | Spicy Stir-Fried Meat | Spicy Duck Jerky | 'Spicy Duck Jerky', 'Spiced Dried Meat', 'Dried Yak Meat' | 0.759 | SEMANTIC |
| 167 | Durian | Frozen Musang King Durian Pulp | 'Frozen Musang King Durian Pulp', 'Frozen Durian', 'Freeze-Dried Durian' | 0.824 | SEMANTIC |
| 168 | Sushi Platter | Sushi and Sashimi Bowl | 'Sushi and Sashimi Bowl', 'Salmon Sashimi Bowl', 'Sushi Platter with Salad' | 0.850 | SEMANTIC |
| 169 | Garlic Crawfish | Baked Lobster with Breadcrumbs | 'Baked Lobster with Breadcrumbs', 'Crawfish with Garlic Sauce', 'Crawfish Dish' | 0.718 | SEMANTIC |
| 170 | Egg Tart | Macau Egg Tart | 'Macau Egg Tart', 'Macau Egg Tart and Pastries', 'Savory Egg Tart' | 0.943 | SEMANTIC |
| 171 | Meat and Corn Porridge | Chicken Porridge with Corn | 'Chicken Porridge with Corn', 'Pork Soup with Corn', 'Pork and Corn Stew' | 0.819 | SEMANTIC |
| 172 | Vegetable Porridge | Corn Soup | 'Corn Soup', 'Yellow Porridge', 'Canned Corn Soup' | 0.778 | SEMANTIC |
| 173 | Chicken Stew | Cat food pate | 'Cat food pate', 'Peanut Stew', 'Paneer Butter Masala' | 0.763 | SEMANTIC |
| 174 | Vegetable Curry with Chicken | Topokki Stew | 'Topokki Stew', 'Japanese Curry', 'Mixed Vegetable and Tofu Soup' | 0.775 | SEMANTIC |
| 175 | Sweet Rolls | Dried Mangoes | 'Dried Mangoes', 'Packaged dumplings', 'Packaged Bread' | 0.653 | MISS |
| 176 | Matcha Latte with Tapioca Pearls | Layered Green Tea Beverage | 'Layered Green Tea Beverage', 'Layered Green Tea Drink', 'Mung Bean Drink' | 0.749 | SEMANTIC |
| 177 | Chicken Soup | Samgyetang (Korean Ginseng Chicken Soup) | 'Samgyetang (Korean Ginseng Chicken Soup)', 'Samgyetang (Ginseng Chicken Soup)', 'Boiled Chicken with Soup' | 0.667 | MISS |
| 178 | Grilled Skewers with Vegetables | Skewered Meat and Fried Snacks | 'Skewered Meat and Fried Snacks', 'Skewered Street Food', 'Assorted Edible Insects' | 0.774 | SEMANTIC |
| 179 | Braised Pork | Roasted Meat with Lettuce | 'Roasted Meat with Lettuce', 'Roast Duck with Meat Mixture', 'Steamed Cabbage with Pork' | 0.803 | SEMANTIC |
| 180 | Omelette with Rice | Vietnamese Omelet (Bánh Xèo) | 'Vietnamese Omelet (Bánh Xèo)', 'Chicken Omelette with Rice', 'Vietnamese Omelette' | 0.754 | SEMANTIC |
| 181 | Braised Pork Hocks | Braised Chicken with Green Onions | 'Braised Chicken with Green Onions', 'Braised Meat with Green Onions', 'Spicy Frog Legs Stir-fry' | 0.808 | SEMANTIC |
| 182 | Cucumber Salad | Stir-fried Cucumbers | 'Stir-fried Cucumbers', 'Stir-fried cucumber with meat', 'Stir-fried Cucumber with Meat' | 0.850 | SEMANTIC |
| 183 | Grilled Sausages with Vegetables | Stir-fried Insects with Sausages | 'Stir-fried Insects with Sausages', 'Stir-fried Sausages with Green Peppers', 'Stir-fried sausage with green peppers' | 0.762 | SEMANTIC |
| 184 | Mixed Fried Platter | Mixed Plate with Meat and Fried Items | 'Mixed Plate with Meat and Fried Items', 'Mixed platter with fried snacks', 'Mixed Skewers and Fried Snacks' | 0.887 | SEMANTIC |
| 185 | Vegetable Fried Rice | Spicy Rice Dish with Egg | 'Spicy Rice Dish with Egg', 'Egg Fried Rice with Meat', 'Fried Rice with Meat and Egg' | 0.823 | SEMANTIC |
| 186 | Fried Chicken | Fried Chicken Wings with Sesame | 'Fried Chicken Wings with Sesame', 'Fried Chicken with Kimchi', 'Fried Chicken with Crackers' | 0.817 | SEMANTIC |
| 187 | Seafood Stir-fry with Rice | Rice porridge with greens | 'Rice porridge with greens', 'Korean meal with soup and side dishes', 'Porridge with side dishes' | 0.700 | SEMANTIC |
| 188 | Scrambled Eggs with Cucumbers | Scrambled Eggs with Cucumbers | 'Scrambled Eggs with Cucumbers', 'Scrambled Eggs with Green Peppers', 'Cucumber and Scrambled Eggs' | 1.000 | EXACT |
| 189 | Beef Noodle Soup | Beef Noodle Soup with Fried Items | 'Beef Noodle Soup with Fried Items', 'Bún Bò Huế', 'Beef and Daikon Soup' | 0.952 | SEMANTIC |
| 190 | Spring Rolls | Rolled Crepes | 'Rolled Crepes', 'Sweet Bean Paste Rolls', 'Peking Duck Wraps' | 0.759 | SEMANTIC |
| 191 | Steamed Buns | Beef Bao | 'Beef Bao', 'Steamed Buns with Meat', 'Steamed buns with sliced meat' | 0.835 | SEMANTIC |
| 192 | Shrimp and Noodles | Spicy Noodles with Skewers | 'Spicy Noodles with Skewers', 'Noodle platter with sauces', 'Skewered Meat with Noodles' | 0.814 | SEMANTIC |
| 193 | Gourmet Shrimp Salad | Shrimp with Green Puree | 'Shrimp with Green Puree', 'Decorative seafood dish', 'Lobster with Lettuce' | 0.779 | SEMANTIC |
| 194 | Pan-fried Fish | Steamed Fish with Chicken | 'Steamed Fish with Chicken', 'Stir-fried Fish with Herbs', 'Steamed Fish in Soy Sauce' | 0.864 | SEMANTIC |
| 195 | Sausages with Tomatoes | Sliced Sausage Platter | 'Sliced Sausage Platter', 'Sausages with dipping sauce', 'Sausages with Salad' | 0.808 | SEMANTIC |
| 196 | Birthday Cake | Character Cake | 'Character Cake', 'Zhui Mu Cake', 'Preserved Confection' | 0.902 | SEMANTIC |
| 197 | Spicy Crab | Spicy Stir-Fried Frog Legs | 'Spicy Stir-Fried Frog Legs', 'Stir-fried Shrimp with Chives', 'Stir-fried shellfish' | 0.759 | SEMANTIC |
| 198 | Bacon with Asparagus | Bacon with Asparagus | 'Bacon with Asparagus', 'Bacon and Asparagus', 'Chinese Sausage with Green Onions' | 1.000 | EXACT |
| 199 | Bread | Steam stuffed bun | 'Steam stuffed bun', 'Pan-fried bun', 'Steamed Whole Grain Buns' | 0.774 | SEMANTIC |
| 200 | Crawfish Boil | Crawfish with Green Onions | 'Crawfish with Green Onions', 'Crawfish with Garlic and Green Onions', 'Crawfish with herbs' | 0.764 | SEMANTIC |
| 201 | Snack | Ginseng Candy | 'Ginseng Candy', 'Sagh Tablet Candy', 'White Rabbit Candy' | 0.784 | SEMANTIC |
| 202 | Pears | Asian Pears | 'Asian Pears', 'Sliced Pears', 'Cooked Pears' | 0.945 | SEMANTIC |
| 203 | Pasta with Chicken and Cream Sauce | Chicken Khao Soi | 'Chicken Khao Soi', 'Noodles with Peanut Sauce and Fried Dishes', 'Noodles with curry and fried snacks' | 0.643 | MISS |
| 204 | Buns | Red Bean Bun | 'Red Bean Bun', 'Decorative Cream Buns', 'Sweet Red Bean Bun' | 0.781 | SEMANTIC |
| 205 | Sunflower Seeds | Lan Hua Dou (Peanut Snack) | 'Lan Hua Dou (Peanut Snack)', 'Ginseng Candy', 'Bear-shaped rice treats' | 0.690 | MISS |
| 206 | Dumplings | Dumplings with Noodles | 'Dumplings with Noodles', 'Creamy Dumplings', 'Dumplings with noodles and side dish' | 0.940 | SEMANTIC |
| 207 | Noodle Stir Fry | Noodles with Ham | 'Noodles with Ham', 'Pasta with Ham', 'Heart-Shaped Noodle Dish' | 0.826 | SEMANTIC |
| 208 | Spicy Crawfish | Crawfish and assorted grilled dishes | 'Crawfish and assorted grilled dishes', 'Spicy Crawfish and Side Dishes', 'Crawfish and assorted dishes' | 0.867 | SEMANTIC |
| 209 | Egg Curry | Coconut Chicken Curry | 'Coconut Chicken Curry', 'Fish Curry with Rice', 'Paneer Butter Masala' | 0.859 | SEMANTIC |
| 210 | Dried Vegetables | Preserved Meat | 'Preserved Meat', 'Purple Sweet Potato', 'Dried Red Dates' | 0.793 | SEMANTIC |
| 211 | Noodle Bowl with Pork | Vietnamese Rice Noodle with Roasted Pork | 'Vietnamese Rice Noodle with Roasted Pork', 'Vietnamese Noodle Bowl with Spring Rolls', 'Vietnamese Rice Noodle Bowl' | 0.846 | SEMANTIC |
| 212 | Pasta with Fish | Canned White Beans in Tomato Sauce | 'Canned White Beans in Tomato Sauce', 'Spicy Chicken and Bean Dish', 'Pork with Beans' | 0.728 | SEMANTIC |
| 213 | Stir-fried Beef with Vegetables | Stir-fried Beef with Peppers and Vegetables | 'Stir-fried Beef with Peppers and Vegetables', 'Spicy Beef with Vegetables', 'Stir-fried beef with peppers' | 0.961 | SEMANTIC |
| 214 | Fried Dough Balls | Korean Jeon (Pancakes) | 'Korean Jeon (Pancakes)', 'Pan-fried buns', 'English Muffins' | 0.680 | MISS |
| 215 | Cheeseburger | Cheeseburger Bagel | 'Cheeseburger Bagel', 'Beef Bagel Sandwich', 'Bagel Burger' | 0.844 | SEMANTIC |
| 216 | Grain Bowl with Cucumber | Mango Quinoa | 'Mango Quinoa', 'Baby Nutritional Porridge', 'Frozen Mango' | 0.630 | MISS |
| 217 | Spicy Chicken Stew | Spicy Sichuan Cuisine | 'Spicy Sichuan Cuisine', 'Spicy Pork Soup', 'Spicy Meat Soup and Stir-Fried Dish' | 0.797 | SEMANTIC |
| 218 | Peaches | Peaches and Nectarines | 'Peaches and Nectarines', 'Fresh Peaches', 'Canned Yellow Peaches' | 0.916 | SEMANTIC |
| 219 | Stuffed Flatbread | Baked Calzone | 'Baked Calzone', 'Calzone', 'Meat-filled flatbread' | 0.773 | SEMANTIC |
| 220 | Fried Rice Cake | Sesame Rice Crackers | 'Sesame Rice Crackers', 'Golden Rice Crackers', 'Savory Rice Crackers' | 0.753 | SEMANTIC |
| 221 | Spicy Noodle Soup | Spicy Noodle Soup with Fried Items | 'Spicy Noodle Soup with Fried Items', 'Noodle Soup with Fried Items', 'Noodle Soup with Fried Item' | 0.943 | SEMANTIC |
| 222 | Tomato and Scrambled Eggs | Scrambled Eggs with Tomatoes and Rice | 'Scrambled Eggs with Tomatoes and Rice', 'Scrambled Eggs with Ham and Tomatoes', 'Scrambled Eggs with Tomatoes and Cheese' | 0.927 | SEMANTIC |
| 223 | Sweet Bread | Decorated Croissant | 'Decorated Croissant', 'Butterscotch Raisin Bread', 'Croissant and Pastry' | 0.780 | SEMANTIC |
| 224 | Braised Meat | Braised Pork Ribs with Eggs | 'Braised Pork Ribs with Eggs', 'Braised Pork with Bamboo Shoots', 'Braised Pork with Pumpkin' | 0.797 | SEMANTIC |
| 225 | Stir-fried Bamboo Shoots with Pork | Stir-fried Bamboo Shoots with Meat | 'Stir-fried Bamboo Shoots with Meat', 'Stir-fried bamboo shoots with pork', 'Stir-fried Bamboo Shoots with Pork' | 0.975 | SEMANTIC |
| 226 | Stir-fried Eggplant with Chili | Spicy Chicken with Dried Red Chilies | 'Spicy Chicken with Dried Red Chilies', 'Stir-fried Eggplant with Chicken', 'Spicy Stir-Fried Eggplant' | 0.729 | SEMANTIC |
| 227 | Crispy Pork Belly | Fried Sweet Glazed Bites | 'Fried Sweet Glazed Bites', 'BBQ Chicken Drumsticks', 'Roast Duck with Sauce' | 0.722 | SEMANTIC |
| 228 | Organic Oat Milk | Dried Edible Insects | 'Dried Edible Insects', 'Oat Flour', 'Packaged Rice Noodles' | 0.752 | SEMANTIC |
| 229 | Durian with accompaniments | Steamed Fish in Banana Leaves | 'Steamed Fish in Banana Leaves', 'Leaf-wrapped rice and cake', 'Zongzi (Rice Dumpling)' | 0.652 | MISS |
| 230 | Noodle Soup with Fried Chicken | Dim Sum and Noodle Soup | 'Dim Sum and Noodle Soup', 'Dim Sum and Noodles', 'Noodles with soup and donut' | 0.833 | SEMANTIC |
| 231 | Iced Tea | Mango Passion Fruit Frappe | 'Mango Passion Fruit Frappe', 'Orange Drink', 'Mango Bubble Tea' | 0.720 | SEMANTIC |
| 232 | Strawberries | Strawberry | 'Strawberry', 'Yan Yan Strawberry Snack', 'Strawberry Snack' | 0.950 | SEMANTIC |
| 233 | Mixed Salad | Steamed Fish with Noodles and Vegetables | 'Steamed Fish with Noodles and Vegetables', 'Noodle Salad with Crispy Pancake', 'Steamed Fish with Vegetables and Noodles' | 0.647 | MISS |
| 234 | Meatball Soup | Sweet Rice Balls in Ginger Syrup | 'Sweet Rice Balls in Ginger Syrup', 'Sweet Rice Balls in Soup', 'Mung Bean Soup with Glutinous Rice Balls' | 0.693 | MISS |
| 235 | Fruit-flavored snack | Jiao Yan Su Biscuits | 'Jiao Yan Su Biscuits', 'Lan Hua Dou (Peanut Snack)', 'Macau-style snack' | 0.747 | SEMANTIC |
| 236 | Baked Casserole | Spanish Tortilla | 'Spanish Tortilla', 'Cheesy Corn Bake', 'Moussaka' | 0.773 | SEMANTIC |
| 237 | Stir-fried Noodles | Stir-fried Rice Cakes with Meat | 'Stir-fried Rice Cakes with Meat', 'Stir-fried Octopus', 'Stir-fried Octopus and Chicken' | 0.858 | SEMANTIC |
| 238 | Noodle with Beef | Meat and Egg Noodle Dish | 'Meat and Egg Noodle Dish', 'Spicy Meat Platter with Noodles', 'Noodle dish with assorted meats' | 0.883 | SEMANTIC |
| 239 | Grilled Meat Skewers | Grilled Chicken Wings and Kebab | 'Grilled Chicken Wings and Kebab', 'Fried Meat Skewers', 'Grilled Chicken and Skewers' | 0.828 | SEMANTIC |
| 240 | Vegetable Stew with Dumplings | Pounded Yam with Vegetable Sauce | 'Pounded Yam with Vegetable Sauce', 'Pounded Yam with Vegetable Soup', 'Vegetable and Fish Stew' | 0.738 | SEMANTIC |
| 241 | Buckwheat with Butter and Tomatoes | Buckwheat with Butter and Tomatoes | 'Buckwheat with Butter and Tomatoes', 'Bulgur with beans and tomatoes', 'Bulgur Wheat' | 1.000 | EXACT |
| 242 | Fried Rice | Stir-fried Rice with Cabbage | 'Stir-fried Rice with Cabbage', 'Fried Rice with Stir-fried Vegetables', 'Fried Rice with Stir-Fried Vegetables' | 0.834 | SEMANTIC |
| 243 | Custard Tarts | Egg Tarts and Flaky Pastries | 'Egg Tarts and Flaky Pastries', 'Savory Egg Tart', 'Macau Egg Tart and Pastries' | 0.876 | SEMANTIC |
| 244 | Fried Chicken Bites | Fried Chicken Bites | 'Fried Chicken Bites', 'Crispy Chicken Bites', 'Fried Cauliflower' | 1.000 | EXACT |
| 245 | Noodle Soup | Spicy Tofu Noodle Soup | 'Spicy Tofu Noodle Soup', 'Vietnamese Noodle Bowl', 'Noodle Bowl with Greens' | 0.800 | SEMANTIC |
| 246 | Meat Bun | Pulled Beef Sandwich | 'Pulled Beef Sandwich', 'Shredded Meat Pastry', 'Shredded Pork Mayonnaise Sandwich' | 0.819 | SEMANTIC |
| 247 | Beef Rice Bowl | Rendang with Rice | 'Rendang with Rice', 'Spicy Beef Rendang with Rice', 'Rendang' | 0.743 | SEMANTIC |
| 248 | Decorative Snowman Dessert | Decorative Snowman Dessert | 'Decorative Snowman Dessert', 'Matcha-flavored dessert', 'Matcha Mousse' | 1.000 | EXACT |
| 249 | Crispy Chicken with Sides | Tasting Plate | 'Tasting Plate', 'Mixed Restaurant Dishes', 'Mixed Restaurant Meal' | 0.777 | SEMANTIC |
| 250 | Grapefruit | Mandarin Orange | 'Mandarin Orange', 'Peeled Mango', 'Peeled Orange' | 0.806 | SEMANTIC |
| 251 | Fried Chicken Sandwich | Spicy Chicken Sandwich | 'Spicy Chicken Sandwich', 'Spicy Chicken Sandwich Meal', 'Fried Chicken Sandwich with Extra Chicken' | 0.915 | SEMANTIC |
| 252 | Spicy Tofu | Spicy Tofu with Legumes | 'Spicy Tofu with Legumes', 'Spicy Tofu with Sauce', 'Spicy Tofu with Meat' | 0.925 | SEMANTIC |
| 253 | Shrimp with Dipping Sauce | Steamed Seafood with Dipping Sauce | 'Steamed Seafood with Dipping Sauce', 'Vietnamese Pancakes with Seafood', 'Steamed Shrimp with Dipping Sauce' | 0.875 | SEMANTIC |
| 254 | Buckwheat with Egg and Bread | Canned Yellowfin Tuna | 'Canned Yellowfin Tuna', 'Tuna and Bean Salad', 'Pistachio-Crusted Salmon' | 0.560 | MISS |
| 255 | Grilled Chicken with Rice | Roasted Chicken with Fried Rice | 'Roasted Chicken with Fried Rice', 'Couscous with Chicken and Carrots', 'Roasted Duck with Fried Rice' | 0.881 | SEMANTIC |
| 256 | Braised Chicken Wings | Braised Chicken Wings | 'Braised Chicken Wings', 'Braised Chicken/ Duck', 'Chicken Adobo' | 1.000 | EXACT |
| 257 | Beef Stir-Fry | Stir-fried Liver with Greens | 'Stir-fried Liver with Greens', 'Stir-fried Beef with Green Vegetables', 'Stir-fried beef with green vegetables' | 0.820 | SEMANTIC |
| 258 | Stir-fried vegetables with rice and clams | Stir-fried vegetables with protein and a side of starchy cake | 'Stir-fried vegetables with protein and a side of starchy cake', 'Stir-fried vegetables with rice and dessert', 'Tomato and Egg Stir-fry with Side Dishes' | 0.758 | SEMANTIC |
| 259 | Grilled Steak | Grilled Steak Platter | 'Grilled Steak Platter', 'Grilled Steak with Black Garlic', 'Grilled Beef Strips' | 0.900 | SEMANTIC |
| 260 | Beef and Vegetable Rice | Beef and Zucchini Rice | 'Beef and Zucchini Rice', 'Beef and Zucchini Rice Bowl', 'Rice with meat and cucumber' | 0.857 | SEMANTIC |
| 261 | Dried Fruits | Dried Date | 'Dried Date', 'Dried Flower Tea', 'Dried Chrysanthemum Tea' | 0.844 | SEMANTIC |
| 262 | Mango Sago | Mango Sago | 'Mango Sago', 'Sweet Soup with Tapioca Balls', 'Mango Jelly Dessert' | 1.000 | EXACT |
| 263 | Cherries | Fresh Cherries | 'Fresh Cherries', 'Red cherries', 'Mixed Cherries' | 0.957 | SEMANTIC |
| 264 | Stir-fried Vegetables with Noodles | Braised Abalone with Pork | 'Braised Abalone with Pork', 'Stir-fried bamboo shoots with pork', 'Stir-fried Bamboo Shoots with Pork' | 0.652 | MISS |
| 265 | Fruit Salad with Ice Cream | Fruit Rice Dessert | 'Fruit Rice Dessert', 'Dessert with Tapioca and Fruits', 'Fruit Salad with Tapioca' | 0.828 | SEMANTIC |
| 266 | Apple | Red Apples | 'Red Apples', 'Raw Apples', 'Apple Snack' | 0.878 | SEMANTIC |
| 267 | Beef and Zucchini Stir-Fry | Beef and Zucchini Rice | 'Beef and Zucchini Rice', 'Beef and Zucchini Stir-Fry', 'Beef and Zucchini Rice Bowl' | 0.891 | SEMANTIC |
| 268 | Chicken and Vegetable Stir-fry | Chicken and Squash Salad | 'Chicken and Squash Salad', 'Chicken and Buckwheat Salad', 'Chicken with Nuts' | 0.761 | SEMANTIC |
| 269 | Pizza | Mixed Seafood and Meat Pizza | 'Mixed Seafood and Meat Pizza', 'Pizza', 'Hawaiian Pizza' | 0.837 | SEMANTIC |
| 270 | Hamburger | German Sausage Double Beef Burger | 'German Sausage Double Beef Burger', 'Burger with Fried Side', 'Street Food Sandwich' | 0.850 | SEMANTIC |
| 271 | Hot Pot | Hot Pot with Sushi and Snacks | 'Hot Pot with Sushi and Snacks', 'Mixed Chinese Seafood and Meat Dishes', 'Sashimi or Raw Seafood Platter' | 0.900 | SEMANTIC |
| 272 | Stir-fried Noodles | Spicy Stir-Fried Noodles and Vegetables | 'Spicy Stir-Fried Noodles and Vegetables', 'Spicy Stir-Fried Noodles', 'Noodle Stir Fry with Fried Item' | 0.922 | SEMANTIC |
| 273 | Seafood Feast | Mixed Seafood and Meat Dishes | 'Mixed Seafood and Meat Dishes', 'Mixed Seafood and Meat Feast', 'Seafood Feast' | 0.909 | SEMANTIC |
| 274 | Cheeseburger | In-N-Out Burger | 'In-N-Out Burger', 'Cheeseburger with Soft Drink', 'Burger with Egg' | 0.728 | SEMANTIC |
| 275 | Hot Pot | Claypot Chicken Rice | 'Claypot Chicken Rice', 'Mushroom Hot Pot', 'Claypot Rice' | 0.646 | MISS |
| 276 | Sashimi Bowl | Sashimi and Fried Food Platter | 'Sashimi and Fried Food Platter', 'Sashimi or Raw Seafood Platter', 'Sashimi and assorted meats' | 0.849 | SEMANTIC |
| 277 | Grilled Fish and Skewers | Yakitori | 'Yakitori', 'Grilled Yakitori', 'Kushikatsu' | 0.725 | SEMANTIC |
| 278 | Sliced Duck | Sliced Pork Belly | 'Sliced Pork Belly', 'Sliced Pork', 'Sliced Roast Meat' | 0.785 | SEMANTIC |
| 279 | Mixed Asian Dishes | Mixed Indonesian Dishes | 'Mixed Indonesian Dishes', 'Mixed Filipino Dishes', 'Chicken Soup and Stir-Fried Eggplant' | 0.926 | SEMANTIC |
| 280 | Grilled Mixed Platter | Fried Chicken Wings and Wrapped Meat | 'Fried Chicken Wings and Wrapped Meat', 'Mixed Meat Platter and Appetizers', 'Skewered Meat and Fried Snacks' | 0.708 | SEMANTIC |
| 281 | Braised Pork Belly | Spicy Tofu with Broccoli | 'Spicy Tofu with Broccoli', 'Stir-fried Tofu with Meat', 'Sweet Soy Glazed Tofu' | 0.677 | MISS |
| 282 | Meat Pizza | Fried Chicken Pizza | 'Fried Chicken Pizza', 'Peking Duck Pizza', 'Meat Lover's Pizza' | 0.813 | SEMANTIC |
| 283 | Apple | Tomato | 'Tomato', 'Fresh Tomato', 'Raw Tomato' | 0.837 | SEMANTIC |
| 284 | Hot Pot | Seafood Claypot | 'Seafood Claypot', 'Kimchi Jjigae', 'Korean Hot Pot' | 0.739 | SEMANTIC |
| 285 | Dumplings | Dumplings with Noodles | 'Dumplings with Noodles', 'Creamy Dumplings', 'Decorative Dumplings' | 0.940 | SEMANTIC |
| 286 | Chicken Stir-Fry with Rice | Stir-fried Pork with Green Peppers and Rice | 'Stir-fried Pork with Green Peppers and Rice', 'Stir-fried mushrooms with green peppers and pork', 'Stir-fried meat with green peppers and rice' | 0.838 | SEMANTIC |
| 287 | Spaghetti and Meatballs | Sausage and Meatballs with Noodles | 'Sausage and Meatballs with Noodles', 'Meatballs with Noodles', 'Spaghetti and Meatballs' | 0.897 | SEMANTIC |
| 288 | Mapo Tofu | Stir-fried Tofu with Ground Meat | 'Stir-fried Tofu with Ground Meat', 'Stir-fried bamboo shoots with pork', 'Stir-fried Bamboo Shoots with Pork' | 0.746 | SEMANTIC |
| 289 | Grilled Meat Platter | Sushi and Grilled Fish Platter | 'Sushi and Grilled Fish Platter', 'Sizzling Meat and Seafood Platter', 'Grilled Skewers and Snacks' | 0.819 | SEMANTIC |
| 290 | Fruit Salad | Grain Bowl with Vegetables and Fruits | 'Grain Bowl with Vegetables and Fruits', 'Fruit and Grain Salad', 'Salad and Fruit Bowl' | 0.761 | SEMANTIC |
| 291 | Vegetable Fried Rice | Vegetable Risotto | 'Vegetable Risotto', 'Spiced Rice with Vegetables', 'Stir-fried rice with vegetables and protein' | 0.839 | SEMANTIC |
| 292 | Beef-flavored chips | Indo Chips | 'Indo Chips', 'Lay's Italian Red Meat Flavor Chips', 'Beef-flavored chips' | 0.847 | SEMANTIC |
| 293 | Grilled Skewers | Grilled meat with vegetables and raw beef tartare | 'Grilled meat with vegetables and raw beef tartare', 'Skewered Meat with Fruits', 'Skewered Meat and Frozen Berries' | 0.712 | SEMANTIC |
| 294 | Stir-fried Chicken Feet | Chicken Feet Stir-Fry | 'Chicken Feet Stir-Fry', 'Chicken Feet Stir Fry', 'Stir-fried beef with vegetables and squid' | 0.946 | SEMANTIC |
| 295 | Mixed Chinese Cuisine | Hot Pot and Assorted Dishes | 'Hot Pot and Assorted Dishes', 'Hot Pot and Side Dishes', 'Grilled Meat and Hot Pot' | 0.815 | SEMANTIC |
| 296 | Meat Soup | Bánh Canh | 'Bánh Canh', 'Steamed Fish with Chicken', 'Bún riêu' | 0.690 | MISS |
| 297 | Chicken Rice Bowl | Mixed Japanese Dishes | 'Mixed Japanese Dishes', 'Assorted Japanese Dishes', 'Fried Rice with Assorted Dishes' | 0.774 | SEMANTIC |
| 298 | Layered Cake with Berries | Crepe Cake | 'Crepe Cake', 'Mille-feuille', 'Decorative Dessert Tart' | 0.754 | SEMANTIC |
| 299 | Ice Cream Cone | Sugar-free Ice Cream | 'Sugar-free Ice Cream', 'Strawberry Ice Cream with Honey', 'Soft-Serve Ice Cream' | 0.870 | SEMANTIC |
| 300 | Fried Chicken | Fried Plantains with Legume Stew | 'Fried Plantains with Legume Stew', 'Fried Meat with Plantain Chips', 'Fried Plantain Fritters' | 0.651 | MISS |
| 301 | Tangyuan (Sweet Rice Balls) | Tangyuan | 'Tangyuan', 'Sweet Soup with Tapioca Balls', 'Sweet Soup with Egg Balls' | 0.926 | SEMANTIC |
| 302 | Fried Snack | Youtiao | 'Youtiao', 'Xiangyoutiao', 'Youtiao with Soy Milk' | 0.756 | SEMANTIC |
| 303 | Sliced Meat | Sliced Sausage Platter | 'Sliced Sausage Platter', 'Grilled Sausage Platter', 'Grilled Sausages and Meat' | 0.839 | SEMANTIC |
| 304 | Stir-fried mixed dishes | Stir-fried greens with pork and side dishes | 'Stir-fried greens with pork and side dishes', 'Stir-fried greens with meat and side dishes', 'Stir-fried beef and chicken feet' | 0.859 | SEMANTIC |
| 305 | Watermelon slices | Watermelon and Melon Salad | 'Watermelon and Melon Salad', 'Watermelon Salad', 'Watermelon and assorted fruits' | 0.837 | SEMANTIC |
| 306 | Fried Chicken Sandwich | KFC Sandwich | 'KFC Sandwich', 'Fried Chicken Sandwich with Fried Chicken', 'Fried Chicken Sandwich with Extra Chicken' | 0.878 | SEMANTIC |
| 307 | Beef Bowl | Chashu Rice Bowl | 'Chashu Rice Bowl', 'Meat Rice Bowl with Egg', 'Beef Bowl with Egg' | 0.760 | SEMANTIC |
| 308 | Steamed Fish with Vegetables | Steamed Fish with Ginger and Scallions | 'Steamed Fish with Ginger and Scallions', 'Steamed Fish with Herbs', 'Steamed Fish with Green Onions' | 0.890 | SEMANTIC |
| 309 | Chicken Soup | Pork Rib Seaweed Soup | 'Pork Rib Seaweed Soup', 'Korean Soup (likely Galbitang)', 'Fish Maw Soup' | 0.751 | SEMANTIC |
| 310 | Chicken Wings | Braised Pork with Dates | 'Braised Pork with Dates', 'Meat with Dates', 'Braised Meat with Garlic' | 0.759 | SEMANTIC |
| 311 | Stir-fried shredded meat | Spicy Shredded Chicken | 'Spicy Shredded Chicken', 'Spicy Shredded Meat', 'Spiced Shredded Meat' | 0.831 | SEMANTIC |
| 312 | Cheesy Chicken Casserole | Cheese Hot Pot | 'Cheese Hot Pot', 'Seafood Chawanmushi', 'Chawanmushi' | 0.749 | SEMANTIC |
| 313 | Ice Cream Cone | Cartoon Character Ice Cream | 'Cartoon Character Ice Cream', 'Bubble Tea Dessert', 'Decorative Ice Cream' | 0.828 | SEMANTIC |
| 314 | Marinated Meat | Spicy Meat Ribs | 'Spicy Meat Ribs', 'Cooked Organ Meat in Sauce', 'Spicy Pork Ribs' | 0.801 | SEMANTIC |
| 315 | Grilled Steak with Black Garlic | Grilled Steak with Black Olive Tapenade | 'Grilled Steak with Black Olive Tapenade', 'Steak with olives and tomatoes', 'Grilled Steak with Black Garlic' | 0.908 | SEMANTIC |
| 316 | Pasta with Cream Sauce | Mango dessert with yogurt and sweet soup | 'Mango dessert with yogurt and sweet soup', 'Mango Yogurt', 'Yogurt with sauces' | 0.705 | SEMANTIC |
| 317 | Mapo Tofu | Savory Tofu with Meat Topping | 'Savory Tofu with Meat Topping', 'Sweet Soy Glazed Tofu', 'Sweet Soy Sauce Tofu' | 0.751 | SEMANTIC |
| 318 | Egg Omelette | Scrambled Eggs with Green Onions | 'Scrambled Eggs with Green Onions', 'Scrambled Eggs with Cassava', 'Scrambled Eggs with Green Peppers' | 0.783 | SEMANTIC |
| 319 | Vietnamese Hot Pot | Vietnamese Soup with Bread | 'Vietnamese Soup with Bread', 'Noodle Soup with Fried Snacks', 'Spicy Noodle Soup with Side Salad' | 0.818 | SEMANTIC |
| 320 | Birthday Cake | Panda Dessert | 'Panda Dessert', 'Taro Cake', 'Sweet Taro Dessert' | 0.782 | SEMANTIC |
| 321 | Instant Noodles with Egg | Noodle Soup with Egg and Cheese | 'Noodle Soup with Egg and Cheese', 'Noodle Soup with Fried Egg', 'Noodle Soup with Egg' | 0.868 | SEMANTIC |
| 322 | Spicy Lobster Stir-fry | Crawfish and assorted grilled dishes | 'Crawfish and assorted grilled dishes', 'Seafood and Seed Platter', 'Crawfish and assorted dishes' | 0.746 | SEMANTIC |
| 323 | Spicy Meat Stir-fry | Meat Stew with Dried Fruits | 'Meat Stew with Dried Fruits', 'Braised Meat with Peppers', 'Braised Meat with Crispy Toppings' | 0.767 | SEMANTIC |
| 324 | Sunflower Seeds | Butterfly Pastry | 'Butterfly Pastry', 'Seeds', 'Decorative Pastry' | 0.695 | MISS |
| 325 | Soft Serve Ice Cream | Spicy Soft Serve | 'Spicy Soft Serve', 'Soft-Serve Ice Cream', 'Soft Serve Ice Cream' | 0.939 | SEMANTIC |
| 326 | Nasi Lemak | Mixed Indonesian Rice Plate | 'Mixed Indonesian Rice Plate', 'Nasi Lemak', 'Nasi Lemak and Fried Noodles' | 0.707 | SEMANTIC |
| 327 | Mango Pulp | Peach-flavored jelly | 'Peach-flavored jelly', 'Mango-flavored snack', 'Fruit-flavored gelatin candy' | 0.786 | SEMANTIC |
| 328 | Steamed Bun | Frozen Musang King Durian Pulp | 'Frozen Musang King Durian Pulp', 'Frozen Durian', 'Boiled Egg and Wrapped Food' | 0.576 | MISS |
| 329 | Sweet and Sour Pork | Sweet and Sour Pork Ribs | 'Sweet and Sour Pork Ribs', 'Sweet Soy Glazed Chicken', 'Chinese BBQ Pork (Char Siu)' | 0.919 | SEMANTIC |
| 330 | Stir-fried Noodles | Noodles with Okra | 'Noodles with Okra', 'Spicy Noodles and Stir-Fried Noodles', 'Stir-fried Noodles with Beef and Spinach' | 0.779 | SEMANTIC |
| 331 | White Bread | Hand Tear Bread | 'Hand Tear Bread', 'Heart-shaped Bread with Spread', 'Grilled Triangular Bread' | 0.870 | SEMANTIC |
| 332 | Beef Noodle Soup | Noodle with Beef | 'Noodle with Beef', 'Steamed Noodles with Meat', 'Hot Beef Noodles' | 0.901 | SEMANTIC |
| 333 | Apple | Asian Pear | 'Asian Pear', 'Yellow Apple', 'Pear' | 0.825 | SEMANTIC |
| 334 | Stir-fried meat with green peppers and rice | Stir-fried Pork with Green Peppers and Rice | 'Stir-fried Pork with Green Peppers and Rice', 'Pork and Green Pepper Stir-fry', 'Stir-fried Pork with Green Peppers' | 0.968 | SEMANTIC |
| 335 | Mixed Chinese Dishes | Crawfish and assorted grilled dishes | 'Crawfish and assorted grilled dishes', 'Mixed Chinese Seafood and Meat Dishes', 'Crawfish and assorted dishes' | 0.785 | SEMANTIC |
| 336 | Seaweed Salad | Stir-fried Seaweed | 'Stir-fried Seaweed', 'Stewed Black Fungus', 'Seasoned Seaweed (Laver)' | 0.835 | SEMANTIC |
| 337 | Meat Rice Bowl | Claypot Rice with Egg | 'Claypot Rice with Egg', 'Fried Pork Rice Bowl', 'Claypot Rice' | 0.770 | SEMANTIC |
| 338 | Nasi Lemak | Nasi Kandar | 'Nasi Kandar', 'Nasi Lemak', 'Spicy Chicken Platter' | 0.832 | SEMANTIC |
| 339 | Fried Chicken with Rice Soup | Rice Porridge with Dumpling | 'Rice Porridge with Dumpling', 'Seafood Porridge', 'Rice Porridge with Fried Dough' | 0.771 | SEMANTIC |
| 340 | Fried Rice with Chicken | Taiwanese Rice Dishes | 'Taiwanese Rice Dishes', 'Fried Rice with Assorted Dishes', 'Fried Rice with Egg and Stir-fried Meat' | 0.783 | SEMANTIC |
| 341 | Dumplings and Bread | Dumplings with noodles and side dish | 'Dumplings with noodles and side dish', 'Assorted Dumplings and Fish Cakes', 'Dumplings and assorted dishes' | 0.865 | SEMANTIC |
| 342 | Cucumber | Raw Cucumber | 'Raw Cucumber', 'Cucumber slice', 'Sliced Cucumber' | 0.927 | SEMANTIC |
| 343 | Carrot | Raw Carrot | 'Raw Carrot', 'Carrot', 'Raw Carrots' | 0.928 | SEMANTIC |
| 344 | Soup with Rice | Rice with Egg and Soup | 'Rice with Egg and Soup', 'Soup with rice and side dish', 'Korean Soup with Rice' | 0.887 | SEMANTIC |
| 345 | Meat Curry with Bread | Chicken Tagine | 'Chicken Tagine', 'Paneer Butter Masala', 'Marinated Chicken with Spices' | 0.736 | SEMANTIC |
| 346 | Rice Porridge | Mung Bean Soup with Glutinous Rice Balls | 'Mung Bean Soup with Glutinous Rice Balls', 'Rice Porridge with Meat', 'Mung Bean Porridge' | 0.811 | SEMANTIC |
| 347 | Shrimp Stir-Fry | Transparent Noodles | 'Transparent Noodles', 'Fried Noodles with Floss', 'Noodles or similar grain product' | 0.656 | MISS |
| 348 | Green Mango | Green Mangoes | 'Green Mangoes', 'Stir-fried Fava Beans', 'Honey Roasted Macadamias' | 0.887 | SEMANTIC |
| 349 | Fried Chicken and Fries | Fried food with fries | 'Fried food with fries', 'Fried Shrimp and Fries', 'Mixed Fried Platter' | 0.888 | SEMANTIC |
| 350 | Grilled Meat Skewers | Grilled Skewers with Mochi | 'Grilled Skewers with Mochi', 'Grilled Pineapple and Skewered Meat', 'Grilled Meat Cubes' | 0.879 | SEMANTIC |
| 351 | Mixed Breakfast Plate | Mixed Asian Breakfast | 'Mixed Asian Breakfast', 'Stir-fried Green Beans with Protein Balls', 'Mixed Asian Meal with Pastries' | 0.897 | SEMANTIC |
| 352 | Roasted Carrots and Potatoes | Sweet and Sour Potatoes | 'Sweet and Sour Potatoes', 'Spicy Carrot and Potato Salad', 'Roasted Carrots and Potatoes' | 0.782 | SEMANTIC |
| 353 | Scallion Pancakes | Green Onion Crackers | 'Green Onion Crackers', 'Green Onion Soda Crackers', 'Lahmacun' | 0.796 | SEMANTIC |
| 354 | Spicy Noodle Soup | Bún riêu | 'Bún riêu', 'Spicy Fish Soup', 'Beijing Spicy Noodle Soup' | 0.706 | SEMANTIC |
| 355 | Scrambled Eggs with Spinach | Stir-fried Pumpkin and Greens | 'Stir-fried Pumpkin and Greens', 'Creamy Fish with Spinach', 'Stir-fried Spinach with Egg' | 0.670 | MISS |
| 356 | Watermelon slice | Cut Watermelon | 'Cut Watermelon', 'Diced Melon', 'Raw Melon' | 0.895 | SEMANTIC |
| 357 | Garlic Butter Snails | Spicy Stir-Fried Snails with Meat | 'Spicy Stir-Fried Snails with Meat', 'Spicy Stir-Fried Snails', 'Stir-fried Snails' | 0.832 | SEMANTIC |
| 358 | Shrimp in Sauce | Sweet and Sour Shrimp | 'Sweet and Sour Shrimp', 'Steamed Shrimp with Sauce', 'Shrimp or similar seafood dish' | 0.922 | SEMANTIC |
| 359 | Grilled Skewers | Fried Skewered Food | 'Fried Skewered Food', 'Fried Snack on a Stick', 'Mixed Skewers and Fried Dishes' | 0.889 | SEMANTIC |
| 360 | Pizza | Pizza with Fried Items | 'Pizza with Fried Items', 'Bacon and Corn Pizza', 'Mango Pizza' | 0.891 | SEMANTIC |
| 361 | Stir-fried meat with vegetables | Stir-fried greens with pork and side dishes | 'Stir-fried greens with pork and side dishes', 'Stir-fried greens with meat and side dishes', 'Stir-fried Chicken with Black Fungus' | 0.836 | SEMANTIC |
| 362 | Sweets (possibly coconut balls) | Tangyuan (Sweet Rice Balls) | 'Tangyuan (Sweet Rice Balls)', 'Tangyuan (Glutinous Rice Balls)', 'Tangyuan (glutinous rice balls)' | 0.801 | SEMANTIC |
| 363 | Whole Milk | Yuanqi Meal Bag | 'Yuanqi Meal Bag', 'Chou Dou Fu', 'Green Bean Paste' | 0.670 | MISS |
| 364 | Sushi Platter | Sushi platter with noodles and vegetables | 'Sushi platter with noodles and vegetables', 'Sushi and Grilled Dishes', 'Assorted Sushi Platter' | 0.850 | SEMANTIC |
| 365 | Ham Sandwich | Ham and Tomato Sandwich | 'Ham and Tomato Sandwich', 'Club Sandwich', 'Ham and Cheese Sandwich with Salad' | 0.919 | SEMANTIC |
| 366 | Noodles with Okra | Cold Noodles with Cucumber | 'Cold Noodles with Cucumber', 'Noodle Soup with Cucumber Salad', 'Steamed Rice Noodles' | 0.791 | SEMANTIC |
| 367 | Apples | Red Apples | 'Red Apples', 'Raw Apples', 'Apple Snack' | 0.940 | SEMANTIC |
| 368 | Mixed Nuts | Packaged Olives | 'Packaged Olives', 'Potato and Bean Plate', 'Pickled Garlic' | 0.729 | SEMANTIC |
| 369 | Mango Tapioca Pudding | Sweet Soup with Egg Balls | 'Sweet Soup with Egg Balls', 'Mango dessert with yogurt and sweet soup', 'Mango Sago' | 0.703 | SEMANTIC |
| 370 | Porridge with meat | Lanzhou Ramen | 'Lanzhou Ramen', 'Century Egg Congee', 'Rice Porridge with Meat' | 0.689 | MISS |
| 371 | Rice Crackers | Packaged Rice Ball | 'Packaged Rice Ball', 'Packaged Rice Cake', 'Fanggao (wafer cake)' | 0.788 | SEMANTIC |
| 372 | Omelet with Bread and Tomatoes | Scrambled Eggs with Tomatoes and Bread | 'Scrambled Eggs with Tomatoes and Bread', 'Bagel Sandwich with Eggs and Fruits', 'Scrambled Eggs with Ham and Tomatoes' | 0.895 | SEMANTIC |
| 373 | Egg and Meat Dish | Mixed Vegetable and Egg Dish with Corn and Watermelon | 'Mixed Vegetable and Egg Dish with Corn and Watermelon', 'Vegetable Couscous with Eggs', 'Scrambled Eggs with Corn and Buttered Bread' | 0.761 | SEMANTIC |
| 374 | Stir-fried Meat with Corn and Greens | Beef and Corn Salad | 'Beef and Corn Salad', 'Stir-fried Meat with Corn', 'Stir-fried Meat with Corn and Greens' | 0.770 | SEMANTIC |
| 375 | Dried Fruit Snack | Dried Chicken Snack | 'Dried Chicken Snack', 'Fried Chicken with Crackers', 'Chicharrón' | 0.890 | SEMANTIC |
| 376 | Braised Pork Belly | Steamed Pork Belly | 'Steamed Pork Belly', 'Braised Pork Belly', 'Stir-fried Pork Belly' | 0.925 | SEMANTIC |
| 377 | Fruit Cake | Layered Cake with Berries | 'Layered Cake with Berries', 'Fruit Layer Cake', 'Fruit Layered Cake' | 0.711 | SEMANTIC |
| 378 | Stir-fried meat strips | Stir-fried bamboo shoots with pork | 'Stir-fried bamboo shoots with pork', 'Stir-fried Bamboo Shoots with Pork', 'Stir-fried Bamboo Shoots with Meat' | 0.832 | SEMANTIC |
| 379 | Bananas | Spiced Bananas | 'Spiced Bananas', 'Bananas', 'Banana Chips' | 0.874 | SEMANTIC |
| 380 | Pizza | Cheese and Corn Pizza | 'Cheese and Corn Pizza', 'Cheese Pizza with Corn', 'Bacon and Corn Pizza' | 0.753 | SEMANTIC |
| 381 | Mixed Asian Dishes | Mixed Chinese Seafood and Meat Dishes | 'Mixed Chinese Seafood and Meat Dishes', 'Mixed Seafood and Pork Dishes', 'Stir-fried meat and tofu dishes' | 0.903 | SEMANTIC |
| 382 | Peach | Strawberry Cream Mochi | 'Strawberry Cream Mochi', 'Peach and Steamed Bun', 'Raw Melon' | 0.661 | MISS |
| 383 | Apple and Banana | Apple and Banana | 'Apple and Banana', 'Apple and Bananas', 'Snacks with Bananas' | 1.000 | EXACT |
| 384 | Tonkatsu | Kushikatsu | 'Kushikatsu', 'Tonkatsu', 'Pork Cutlet (Tonkatsu)' | 0.885 | SEMANTIC |
| 385 | Braised Squid | Stir-fried Corn with Meat | 'Stir-fried Corn with Meat', 'Fried Corn Snack', 'Steamed Baby Corn with Sauce' | 0.659 | MISS |
| 386 | Beef and Green Pepper Stir-fry | Stir-fried Snap Peas with Pork | 'Stir-fried Snap Peas with Pork', 'Stir-fried Pork with Snow Peas', 'Stir-fried Snow Peas with Pork' | 0.757 | SEMANTIC |
| 387 | Custard Tarts | Savory Egg Tart | 'Savory Egg Tart', 'Macau Egg Tart and Pastries', 'Macau Egg Tart' | 0.784 | SEMANTIC |
| 388 | Grapefruit | Lemon | 'Lemon', 'Raw Mango', 'Frozen Mango' | 0.814 | SEMANTIC |
| 389 | Banana Dessert | Banana-shaped cake | 'Banana-shaped cake', 'Fruit-shaped dessert', 'Cherries and Banana' | 0.905 | SEMANTIC |
| 390 | Stir-fried Chicken with Edamame | Stir-fried Beef with Peanuts | 'Stir-fried Beef with Peanuts', 'Stir-fried Beef with Mushrooms', 'Stir-fried Beef with Garlic and Nuts' | 0.717 | SEMANTIC |
| 391 | Drink Milk | Red Bean Drink | 'Red Bean Drink', 'Canned Coffee', 'Energy Drink' | 0.777 | SEMANTIC |
| 392 | Stir-fried Beef with Vegetables | Beef and Snap Pea Stir-Fry | 'Beef and Snap Pea Stir-Fry', 'Stir-fried beef with green vegetables', 'Stir-fried Beef with Green Vegetables' | 0.818 | SEMANTIC |
| 393 | Pasta with Meat Sauce | Sausages with Dumplings and Bell Peppers | 'Sausages with Dumplings and Bell Peppers', 'Pasta with meat and vegetable soup', 'Pasta with Ground Meat and Chicken Wings' | 0.722 | SEMANTIC |
| 394 | Noodle with Meat | Fried tofu with bean sprouts and meat soup | 'Fried tofu with bean sprouts and meat soup', 'Bò sốt vang', 'Bún chả' | 0.686 | MISS |
| 395 | Dumplings | Pork Dumplings | 'Pork Dumplings', 'Meat Dumplings', 'Beef Dumplings' | 0.937 | SEMANTIC |
| 396 | Mixed Salad | Vegetable Salad with Century Egg | 'Vegetable Salad with Century Egg', 'Hot Pot with Salad', 'Steamed Fish with Green Onions' | 0.747 | SEMANTIC |
| 397 | Noodle Soup with Shrimp | Steamed Vermicelli with Seafood | 'Steamed Vermicelli with Seafood', 'Mixed Seafood and Noodle Dishes', 'Oyster Noodle Soup' | 0.795 | SEMANTIC |
| 398 | Stir-fried rice with vegetables and meat | Sweet and Sour Pork with Rice | 'Sweet and Sour Pork with Rice', 'Mixed Asian Takeout', 'Takeout Asian Dish' | 0.731 | SEMANTIC |
| 399 | Cake | Fanggao (wafer cake) | 'Fanggao (wafer cake)', 'Durian Wafer', 'Durian Milk Cake' | 0.799 | SEMANTIC |
| 400 | Sucking Jelly | Watermelon-flavored chewing gum | 'Watermelon-flavored chewing gum', 'Strawberry-flavored candy', 'Strawberry-flavored snack' | 0.671 | MISS |
| 401 | Fruit Bunny | Rabbit-shaped dessert | 'Rabbit-shaped dessert', 'Fruit Bunny', 'Cute Animal Pastries' | 0.834 | SEMANTIC |
| 402 | Spicy Grilled Corn | Grilled Corn with Sauce | 'Grilled Corn with Sauce', 'Spicy Grilled Corn', 'Grilled Corn' | 0.927 | SEMANTIC |
| 403 | Fruit Salad | Citrus Fruits | 'Citrus Fruits', 'Citrus fruits', 'Mixed Citrus Fruits' | 0.847 | SEMANTIC |
| 404 | Layered Salad | Purple Rice Cake | 'Purple Rice Cake', 'Pink Coconut Dessert', 'Colorful Rice Cakes' | 0.649 | MISS |
| 405 | Peach | Mango Buns | 'Mango Buns', 'Peeled Mango', 'Decorative Rice Balls' | 0.747 | SEMANTIC |
| 406 | Pasta with Meatballs | Meatball with Potatoes | 'Meatball with Potatoes', 'Sausage and Meatballs with Noodles', 'Carbonara' | 0.855 | SEMANTIC |
| 407 | Grilled Steak with Asparagus and Tomatoes | Grilled Steak with Cherry Tomatoes | 'Grilled Steak with Cherry Tomatoes', 'Grilled Steak with Asparagus and Potatoes', 'Ribeye Steak with Roasted Potatoes and Asparagus' | 0.784 | SEMANTIC |
| 408 | Spicy Tofu | Spicy Black Tofu | 'Spicy Black Tofu', 'Spicy Seaweed Salad', 'Spicy Seaweed Snack' | 0.916 | SEMANTIC |
| 409 | Stir-fried Vegetables and Seafood | Stir-fried greens with meat and side dishes | 'Stir-fried greens with meat and side dishes', 'Stir-fried greens with pork and side dishes', 'Stir-fried beef with green vegetables' | 0.828 | SEMANTIC |
| 410 | Braised Fish | Sizzling Beef with Vegetables | 'Sizzling Beef with Vegetables', 'Braised Meat with Green Onions', 'Spicy Braised Duck' | 0.741 | SEMANTIC |
| 411 | Nescafe 3-in-1 Coffee | Nescafe 3-in-1 Coffee | 'Nescafe 3-in-1 Coffee', 'Nescafé Coffee Mix', 'Instant Coffee' | 1.000 | EXACT |
| 412 | Sweet Pastry | Packaged Cookies | 'Packaged Cookies', 'Packaged Rice Noodles', 'Packaged noodles' | 0.828 | SEMANTIC |
| 413 | Fried Dough Sticks | Fried Cassava Fries | 'Fried Cassava Fries', 'Soybean Curd Sticks', 'Sweet Cassava Sticks' | 0.795 | SEMANTIC |
| 414 | Mixed Asian Cuisine | Hot Pot with Sushi and Snacks | 'Hot Pot with Sushi and Snacks', 'Sashimi and Hot Pot', 'Assorted Japanese Dishes' | 0.708 | SEMANTIC |
| 415 | Burger and Fries | Pizza and Burger Meal | 'Pizza and Burger Meal', 'Chole Bhature', 'Themed Pizza and Burger Meal' | 0.798 | SEMANTIC |
| 416 | Crawfish Stir-fry | Steamed Fish and Beef with Tomatoes | 'Steamed Fish and Beef with Tomatoes', 'Baked Dish with Meat Sauce', 'Steamed Fish with Spicy Topping' | 0.650 | MISS |
| 417 | Beef Noodle Soup | Vietnamese Beef Noodle Soup | 'Vietnamese Beef Noodle Soup', 'Jiaomijiao Pickled Pepper Beef Noodle', 'Sour Beef Noodle Soup' | 0.921 | SEMANTIC |
| 418 | Crinkle-Cut Fries | Crinkle-Cut Fries | 'Crinkle-Cut Fries', 'Cheesy Fries', 'Spicy Curly Fries' | 1.000 | EXACT |
| 419 | Mixed Grill Platter | Korean BBQ with side dishes | 'Korean BBQ with side dishes', 'Sizzling Meat and Seafood Platter', 'Korean BBQ side dishes' | 0.723 | SEMANTIC |
| 420 | Mixed Asian Buffet | Mixed Buffet Dishes | 'Mixed Buffet Dishes', 'Assorted Rice Toppings', 'Bhel Puri' | 0.864 | SEMANTIC |
| 421 | Braised Meat | Preserved Meat with Drink | 'Preserved Meat with Drink', 'Hamburger Steak', 'Liver with sauce' | 0.821 | SEMANTIC |
| 422 | Savory Snack Crackers | Whole Grain Crackers | 'Whole Grain Crackers', 'Sesame Crackers', 'Sesame Rice Crackers' | 0.845 | SEMANTIC |
| 423 | Leafy Greens | Raw Water Spinach | 'Raw Water Spinach', 'Steamed Leafy Greens', 'Chopped Cilantro' | 0.749 | SEMANTIC |
| 424 | Ramen | Chashu Pork | 'Chashu Pork', 'Ramen Bowl', 'Chashu Rice Bowl' | 0.727 | SEMANTIC |
| 425 | Grilled Meat Skewers | Dried Meat Slice | 'Dried Meat Slice', 'Mixed Sausage Platter', 'Blood sausage with vegetables' | 0.736 | SEMANTIC |
| 426 | Papaya | Peeled Mango | 'Peeled Mango', 'Raw Squash', 'Yellow fruit (e.g., apricot)' | 0.732 | SEMANTIC |
| 427 | Snack Mix | Mixed Nuts or Snack Mix | 'Mixed Nuts or Snack Mix', 'Assorted Snack Mix', 'Roasted Mixed Nuts' | 0.920 | SEMANTIC |
| 428 | Grilled Steak with Vegetables | Sizzling Beef with Garlic | 'Sizzling Beef with Garlic', 'Steak with Garlic', 'Beef Tenderloin' | 0.742 | SEMANTIC |
| 429 | Fried Fish | Fried Edible Insects | 'Fried Edible Insects', 'Fried Insect Larvae', 'Assorted Edible Insects' | 0.821 | SEMANTIC |
| 430 | Stuffed Eggplant | Eggplant Rolls | 'Eggplant Rolls', 'Steamed Purple Rice Roll', 'Grilled Seaweed Roll' | 0.820 | SEMANTIC |
| 431 | Braised Pork Belly | Sweet Soy Glazed Tofu | 'Sweet Soy Glazed Tofu', 'Steamed Pork Belly', 'Sweet Soy Sauce Tofu' | 0.767 | SEMANTIC |
| 432 | Fried Dough Balls | Fried Sweet Glazed Bites | 'Fried Sweet Glazed Bites', 'Fried Sweet Ball', 'Fried Bread with Glaze' | 0.838 | SEMANTIC |
| 433 | Noodle Soup | Beijing Spicy Noodle Soup | 'Beijing Spicy Noodle Soup', 'Lanzhou Ramen', 'Spicy Meat Noodle Soup' | 0.836 | SEMANTIC |
| 434 | Seafood Platter | Mixed Seafood and Meat Dishes | 'Mixed Seafood and Meat Dishes', 'Mixed Seafood and Appetizers', 'Seafood Feast' | 0.864 | SEMANTIC |
| 435 | Eggs in Tomato Sauce | Tomato Chicken Stew | 'Tomato Chicken Stew', 'Spicy Chicken with Tomato Sauce', 'Dumplings in Spicy Sauce' | 0.852 | SEMANTIC |
| 436 | Creamy Pasta | Spaghetti Aglio e Olio | 'Spaghetti Aglio e Olio', 'Spaghetti alle Vongole', 'Creamy Mushroom Pasta' | 0.746 | SEMANTIC |
| 437 | Stir-fried Beef with Vegetables | Ground meat with vegetables | 'Ground meat with vegetables', 'Ground Meat and Vegetables', 'Vegetable and Lentil Stew' | 0.852 | SEMANTIC |
| 438 | Spicy Pickled Peppers | Packaged Sausage Snack | 'Packaged Sausage Snack', 'Packaged Chicken Snack', 'Mango Gummy Candy' | 0.747 | SEMANTIC |
| 439 | Cheese Pizza | Cheesy Garlic Bread | 'Cheesy Garlic Bread', 'Cheese Pizza and Mixed Bowl', 'Cheese Pizza with Corn' | 0.789 | SEMANTIC |
| 440 | Apple Pie | Fried Chicken Pie | 'Fried Chicken Pie', 'Apple Pie', 'Meat Pie' | 0.842 | SEMANTIC |
| 441 | Beef Salad | Stir-fried Bean Sprouts with Meat | 'Stir-fried Bean Sprouts with Meat', 'Yusheng (Raw Fish Salad)', 'Korean-style mixed salad' | 0.734 | SEMANTIC |
| 442 | Tomato and Egg Stir-fry | Scrambled Eggs with Tomatoes and Rice | 'Scrambled Eggs with Tomatoes and Rice', 'Tomato Scrambled Eggs with Rice', 'Scrambled Eggs with Tomatoes and Cheese' | 0.823 | SEMANTIC |
| 443 | Cooked Shrimp | Steamed Seafood with Dipping Sauce | 'Steamed Seafood with Dipping Sauce', 'Stir-fried Shrimp with Chives', 'Steamed Shrimp with Dipping Sauce' | 0.786 | SEMANTIC |
| 444 | Rambutan | Rambutan | 'Rambutan', 'Lychee or Longan', 'Peeled Lychee' | 1.000 | EXACT |
| 445 | Curry with Rice | Curry with Bear-Shaped Rice | 'Curry with Bear-Shaped Rice', 'Bear-shaped rice dish with pumpkin', 'Fried Meat with Curry Sauce' | 0.835 | SEMANTIC |
| 446 | Mixed Noodle Soup | Stew with Tofu and Fried Dough | 'Stew with Tofu and Fried Dough', 'Braised Meat with Fried Tofu', 'Bak Kut Teh' | 0.736 | SEMANTIC |
| 447 | Raw Beef | Marbled Beef | 'Marbled Beef', 'Marbled Beef Steak', 'Sliced Marbled Beef' | 0.821 | SEMANTIC |
| 448 | Fried Chicken Sandwich | Fried Chicken Sandwich with Extra Chicken | 'Fried Chicken Sandwich with Extra Chicken', 'Breaded Chicken Sandwich', 'Fried Chicken Sandwich with Fried Chicken' | 0.976 | SEMANTIC |
| 449 | Nutritious Cereal | Noodles or similar grain product | 'Noodles or similar grain product', 'Packaged Rice Noodles', 'Packaged Rice Cake' | 0.785 | SEMANTIC |
| 450 | Braised Pork Belly | Stir-fried Pork Belly | 'Stir-fried Pork Belly', 'Steamed Pork Belly', 'Stir-fried Pork Belly with Scallions' | 0.907 | SEMANTIC |
| 451 | Ice Cream Cone | Kinder Bueno Ice Cream Cone | 'Kinder Bueno Ice Cream Cone', 'Soft-Serve Ice Cream Cone', 'Moutai Ice Cream' | 0.641 | MISS |
| 452 | Braised Chicken | Spicy Braised Pork Ribs | 'Spicy Braised Pork Ribs', 'Braised Pork Ribs with Rice', 'Spicy Braised Ribs' | 0.805 | SEMANTIC |
| 453 | Shrimp and Clam Stir-Fry | Spicy Seafood Stir-fry | 'Spicy Seafood Stir-fry', 'Spicy Stir-Fried Seafood', 'Stir-fried seafood with sauce' | 0.887 | SEMANTIC |
| 454 | Stir-fried Eggplant and Peppers | Stir-fried Squid with Peppers | 'Stir-fried Squid with Peppers', 'Stir-fried Mushrooms with Peppers', 'Stir-fried Eggplant with Peppers' | 0.760 | SEMANTIC |
| 455 | Glazed Chicken Wings | Grilled Chicken Wings with Orange | 'Grilled Chicken Wings with Orange', 'Lemon Chicken Wings', 'Lemon Garlic Chicken Wings' | 0.877 | SEMANTIC |
| 456 | Noodle Salad | Shredded Mango Salad | 'Shredded Mango Salad', 'Shredded Carrot Salad', 'Mango Salad' | 0.728 | SEMANTIC |
| 457 | Mixed Meat Rice Bowl | Chashu Rice Bowl | 'Chashu Rice Bowl', 'Mixed Meat Rice Box', 'Mixed Meat and Rice Plate' | 0.833 | SEMANTIC |
| 458 | Garlic Shrimp | Crawfish with herbs | 'Crawfish with herbs', 'Stir-fried Crawfish', 'Boiled Shrimp' | 0.778 | SEMANTIC |
| 459 | Breaded Eggplant | Baked Zucchini with Cheese | 'Baked Zucchini with Cheese', 'Seafood Corn Patties', 'Green Beans with Cornmeal Cakes' | 0.626 | MISS |
| 460 | Lychee | Salak (Snake Fruit) | 'Salak (Snake Fruit)', 'Dried Lychee', 'Fruit (possibly a type of lychee or similar)' | 0.676 | MISS |
| 461 | Rice Porridge | Oat Yogurt | 'Oat Yogurt', 'Osmanthus Oolong Milk Tea', 'Low-Fat Yogurt Drink' | 0.815 | SEMANTIC |
| 462 | Fish Soup | Century Egg Congee | 'Century Egg Congee', 'Seafood Congee', 'Seafood Rice Porridge' | 0.666 | MISS |
| 463 | Stir-fried Noodles | Stir-fried Enoki Mushrooms with Green Vegetables | 'Stir-fried Enoki Mushrooms with Green Vegetables', 'Stir-fried Glass Noodles', 'Sesame Oil Noodles' | 0.823 | SEMANTIC |
| 464 | Noodle Soup | Vietnamese Rice Noodle Salad | 'Vietnamese Rice Noodle Salad', 'Thai Papaya Salad', 'Vietnamese Noodle Salad' | 0.755 | SEMANTIC |
| 465 | Hot Pot | Hot Pot with Sliced Meats | 'Hot Pot with Sliced Meats', 'Hot Pot with Sliced Meat', 'Korean BBQ with side dishes' | 0.863 | SEMANTIC |
| 466 | Crawfish | Crawfish with Dipping Sauce | 'Crawfish with Dipping Sauce', 'Crawfish with Garlic', 'Boiled Crawfish' | 0.875 | SEMANTIC |
| 467 | Roasted Chestnuts | Roasted Chestnuts | 'Roasted Chestnuts', 'Sweet Chestnuts', 'Chestnuts' | 1.000 | EXACT |
| 468 | Steamed Bun | Pig-shaped steamed buns | 'Pig-shaped steamed buns', 'Colorful Steamed Buns', 'Decorative Steamed Buns' | 0.827 | SEMANTIC |
| 469 | Watermelon | Watermelon Ice Cream Bowl | 'Watermelon Ice Cream Bowl', 'Watermelon Drink', 'Cut Watermelon' | 0.719 | SEMANTIC |
| 470 | Steamed Fish | Steamed Fish with Glass Noodles | 'Steamed Fish with Glass Noodles', 'Steamed Fish in Soy Sauce', 'Steamed Fish with Herbs' | 0.881 | SEMANTIC |
| 471 | Teriyaki Chicken | Sweet Soy Chicken Wings | 'Sweet Soy Chicken Wings', 'Glazed Chicken Wings', 'Fried Chicken Wings with Sesame' | 0.837 | SEMANTIC |
| 472 | Stir-fried vegetables with meat | Stir-fried greens with meat and side dishes | 'Stir-fried greens with meat and side dishes', 'Stir-fried greens with pork and side dishes', 'Stir-fried Greens with Fried Dough' | 0.856 | SEMANTIC |
| 473 | Garlic Shrimp | Stir-fried Shrimp with Peppers | 'Stir-fried Shrimp with Peppers', 'Spicy Fried Seafood', 'Spicy Shrimp and Chicken Stir-fry' | 0.810 | SEMANTIC |
| 474 | Green Apple | Green Apple | 'Green Apple', 'Green Apples', 'Green Apple with Bread' | 1.000 | EXACT |
| 475 | Pineapple Beef Stir-Fry | Braised Meat with Pineapple | 'Braised Meat with Pineapple', 'Meatballs with Pineapple', 'Beef Stew with Pineapple' | 0.834 | SEMANTIC |
| 476 | Mixed Fried Snacks | Mixed platter with fried snacks | 'Mixed platter with fried snacks', 'Mixed Vegetable and Fried Items Platter', 'Mixed Vegetable and Fried Snack Platter' | 0.883 | SEMANTIC |
| 477 | Barbecued Ribs | Roast Meat with Corn and Potatoes | 'Roast Meat with Corn and Potatoes', 'Beef Ribs with Vegetables', 'Pineapple Pork Ribs' | 0.731 | SEMANTIC |
| 478 | Sushi | Sushi and Grilled Dishes | 'Sushi and Grilled Dishes', 'Sushi and assorted Japanese dishes', 'Sushi and Asian Cuisine Platter' | 0.869 | SEMANTIC |
| 479 | Colorful Jelly Desserts | Steamed Meatballs with Dipping Sauce | 'Steamed Meatballs with Dipping Sauce', 'Gourmet Canapés', 'Creamy Canapés' | 0.628 | MISS |
| 480 | Mixed Chinese Dishes | Korean multi-course meal | 'Korean multi-course meal', 'Shanxi Fried Pork', 'Weilong Hot Strips' | 0.873 | SEMANTIC |
| 481 | Pork Belly Rice Bowl | Pork Belly with Eggs | 'Pork Belly with Eggs', 'Braised Pork Belly with Eggs', 'Pork Slices with Egg' | 0.798 | SEMANTIC |
| 482 | Stuffed Pancakes | Dried Codfish Fillet | 'Dried Codfish Fillet', 'Raw Fish Fillets', 'Pan-fried Fish' | 0.659 | MISS |
| 483 | Barbecue Ribs | Spicy Glazed Ribs | 'Spicy Glazed Ribs', 'Spicy Pork Ribs', 'Barbecue Pork Ribs' | 0.919 | SEMANTIC |
| 484 | Durian | Frozen Musang King Durian Pulp | 'Frozen Musang King Durian Pulp', 'Freeze-Dried Durian', 'Frozen Durian' | 0.824 | SEMANTIC |
| 485 | Hot Dog | Hot Dog Wrap | 'Hot Dog Wrap', 'Hot Dog Wraps', 'Loaded Hot Dog' | 0.881 | SEMANTIC |
| 486 | Noodle Soup with Chicken | Korean Soup (likely Galbitang) | 'Korean Soup (likely Galbitang)', 'Naengmyeon', 'Korean meal with soup and side dishes' | 0.825 | SEMANTIC |
| 487 | Sweet and Sour Wontons | Honey Garlic Chicken Wings | 'Honey Garlic Chicken Wings', 'Orange Chicken', 'Honey Garlic Chicken' | 0.731 | SEMANTIC |
| 488 | Chicken Stir-Fry | Stir-fried Chicken with Green Peppers | 'Stir-fried Chicken with Green Peppers', 'Stir-fried Chicken with Snow Peas', 'Stir-fried Green Peppers with Chicken' | 0.853 | SEMANTIC |
| 489 | Green Apple | Fruit and Vegetable Snack Box | 'Fruit and Vegetable Snack Box', 'Stuffed Bitter Melon', 'Fresh Melons' | 0.713 | SEMANTIC |
| 490 | Cheesecake | Kashkaval Cheese | 'Kashkaval Cheese', 'Mung Bean Cake', 'Fruit Cake Slice' | 0.771 | SEMANTIC |
| 491 | Waffles | Waffle Snack | 'Waffle Snack', 'Peanut Waffle Snack', 'Waffles with Syrup' | 0.937 | SEMANTIC |
| 492 | Noodle Soup | Tom Yum Noodles | 'Tom Yum Noodles', 'Spicy Seafood Noodles', 'Noodle Dish with Cracklings' | 0.866 | SEMANTIC |
| 493 | Stir-fried Green Vegetables | Stir-fried Water Spinach | 'Stir-fried Water Spinach', 'Stir-fried greens with protein', 'Stir-fried Greens with Garlic' | 0.789 | SEMANTIC |
| 494 | Raw Eggplant | Eggplant | 'Eggplant', 'Sweet and Sour Eggplant', 'Raw Eggplant' | 0.917 | SEMANTIC |
| 495 | Chicken Noodle Dish | Grilled Fish with Noodles | 'Grilled Fish with Noodles', 'Grilled Fish with Vegetables and Noodles', 'Grilled Chicken with Noodles' | 0.817 | SEMANTIC |
| 496 | Dumplings | Dumplings with Spicy Sauce | 'Dumplings with Spicy Sauce', 'Sweet Dumplings in Ginger Syrup', 'Steamed Dumplings with Sauce' | 0.902 | SEMANTIC |
| 497 | Mixed Meat Platter | Mixed Salad Plate with Tuna and Turkey | 'Mixed Salad Plate with Tuna and Turkey', 'Sliced Pork with Side Dishes', 'Mixed Protein Platter' | 0.796 | SEMANTIC |
| 498 | Stone Bowl Dish | Century Egg Congee | 'Century Egg Congee', 'Steamed Rice with Pickles', 'Matcha dessert with red bean' | 0.606 | MISS |
| 499 | Stir-fried Tofu | Fried Tofu with Green Onions | 'Fried Tofu with Green Onions', 'Fried Tofu Cubes', 'Fried Tofu with Chilies' | 0.858 | SEMANTIC |
| 500 | French Fries | Potato Wedges with Ketchup | 'Potato Wedges with Ketchup', 'Baked Potato Wedges with Ketchup', 'Crinkle-Cut Fries with Dipping Sauces' | 0.726 | SEMANTIC |
| 501 | Grilled Steak | Beef Tenderloin | 'Beef Tenderloin', 'Raw Ribeye Steak', 'Pan-Seared Steak' | 0.856 | SEMANTIC |
| 502 | Stir-fried noodles with rice | Grilled Meat with Rice and Salad | 'Grilled Meat with Rice and Salad', 'Mixed Rice Plate with Beef', 'Grilled Meat with Rice and Vegetables' | 0.721 | SEMANTIC |
| 503 | Noodle Salad | Vietnamese Rice Noodle with Roasted Pork | 'Vietnamese Rice Noodle with Roasted Pork', 'Vietnamese Rice Noodle Platter', 'Vietnamese noodle platter' | 0.750 | SEMANTIC |
| 504 | Shrimp Noodles | Shrimp Noodle Stir-Fry | 'Shrimp Noodle Stir-Fry', 'Shrimp Noodle Stir-fry', 'Shrimp and Noodle Stir-Fry' | 0.903 | SEMANTIC |
| 505 | Roasted Coffee Beans | Raw Coffee Beans | 'Raw Coffee Beans', 'Salted Black Soybeans', 'Roasted Coffee Beans' | 0.926 | SEMANTIC |
| 506 | Grapes | Seedless Grapes | 'Seedless Grapes', 'Kyoho Grapes', 'Grapes' | 0.885 | SEMANTIC |
| 507 | Dairy Drink | Fermented Milk Drink | 'Fermented Milk Drink', 'Apple Milk Drink', 'Apple-flavored milk' | 0.913 | SEMANTIC |
| 508 | Steamed Buns | Steamed Dough Buns | 'Steamed Dough Buns', 'Pig-shaped steamed buns', 'Steamed Whole Grain Buns' | 0.956 | SEMANTIC |
| 509 | Steamed Egg | Packaged Rice Ball | 'Packaged Rice Ball', 'Lychee-flavored snack', 'Ginseng Candy' | 0.703 | SEMANTIC |
| 510 | Bananas | Snacks with Bananas | 'Snacks with Bananas', 'Bananas', 'Banana' | 0.905 | SEMANTIC |
| 511 | Vegetable Noodles | Stir-fried vegetables with spaghetti | 'Stir-fried vegetables with spaghetti', 'Vegetable Lo Mein', 'Spicy Noodles with Stir-fried Vegetables' | 0.847 | SEMANTIC |
| 512 | Dried Prunes | Dried Prunes | 'Dried Prunes', 'California Prunes', 'Prunes' | 1.000 | EXACT |
| 513 | Beef and Corn Stir-fry | Beef and Corn Soup | 'Beef and Corn Soup', 'Beef and Corn Stew', 'Meat Stew with Corn' | 0.822 | SEMANTIC |
| 514 | Hot Pot | Hot Pot with Sliced Meats | 'Hot Pot with Sliced Meats', 'Korean Hot Pot', 'Hot Pot with Assorted Meats' | 0.863 | SEMANTIC |
| 515 | Pork and Green Beans Stir-fry | Stir-fried Pork with Green Beans | 'Stir-fried Pork with Green Beans', 'Stir-fried Meat with Green Beans', 'Stir-fried meat with green beans' | 0.925 | SEMANTIC |
| 516 | Baked Chicken Wings | Packaged Cooked Chicken | 'Packaged Cooked Chicken', 'Grilled Chicken Wings with Orange', 'Packaged Chicken Snack' | 0.791 | SEMANTIC |
| 517 | Bananas | Green Bananas | 'Green Bananas', 'Bananas', 'Banana' | 0.905 | SEMANTIC |
| 518 | Fast Food Meal | Fast Food Combo | 'Fast Food Combo', 'KFC Burger Meal', 'Burgers and Fried Chicken' | 0.949 | SEMANTIC |
| 519 | Egg Sandwich | Packaged Snack Cake | 'Packaged Snack Cake', 'Subway Salad', 'Packaged Egg Snack' | 0.777 | SEMANTIC |
| 520 | Oranges | Canned Mandarin Oranges | 'Canned Mandarin Oranges', 'Mandarin Orange', 'Citrus Fruit Drops' | 0.818 | SEMANTIC |
| 521 | Noodles with Sauce | Noodles with Spicy Sauce | 'Noodles with Spicy Sauce', 'Noodles with Crawfish', 'Noodles with Peanut Sauce' | 0.958 | SEMANTIC |
| 522 | Beef Stir-Fry | Stir-fried beef with vegetables, tomato and egg stir-fry, and sautéed greens | 'Stir-fried beef with vegetables, tomato and egg stir-fry, and sautéed greens', 'Beef and Broccoli with Eggs', 'Braised Pork with Egg and Greens' | 0.749 | SEMANTIC |
| 523 | Noodle Soup with Dumplings | Spicy Soup with Fried Snacks | 'Spicy Soup with Fried Snacks', 'Fried snacks with spicy soup', 'Noodle dishes with accompaniments' | 0.768 | SEMANTIC |
| 524 | Sandwich with salad | Smoked Salmon Sandwich | 'Smoked Salmon Sandwich', 'Salmon Croissant Sandwich', 'Shrimp and Egg Sandwich' | 0.822 | SEMANTIC |
| 525 | Seafood Noodle Soup | Seafood Noodle Soup with Crab | 'Seafood Noodle Soup with Crab', 'Seafood Hot Pot with Noodles', 'Seafood Noodle Hot Pot' | 0.961 | SEMANTIC |
| 526 | Stir-fried beans and meat | Spicy Chicken with Peanuts | 'Spicy Chicken with Peanuts', 'Spicy Chicken with Nuts', 'Sweet and Sour Chicken with Corn' | 0.745 | SEMANTIC |
| 527 | Mixed Rice | Black Sesame Crackers | 'Black Sesame Crackers', 'Black Rice Cake', 'Cookies and Cream Snack' | 0.676 | MISS |
| 528 | Mixed Cuisine Platter | Assorted Banquet Dishes | 'Assorted Banquet Dishes', 'Assorted Vietnamese Dishes', 'Assorted Korean Dishes' | 0.882 | SEMANTIC |
| 529 | Pilaf | Sweet Rice with Dried Fruits | 'Sweet Rice with Dried Fruits', 'Rice with Red Dates', 'Rice with assorted dishes' | 0.730 | SEMANTIC |
| 530 | Stir-fried Mixed Vegetables with Meat | Vegetable Japchae | 'Vegetable Japchae', 'Traditional Chinese Herbal Soup', 'Stir-fried Seaweed and Vegetables' | 0.724 | SEMANTIC |
| 531 | Hot Pot | Hot Pot with Sliced Meats | 'Hot Pot with Sliced Meats', 'Hot Pot with Sliced Meat', 'Sashimi and Hot Pot' | 0.863 | SEMANTIC |
| 532 | Squid Ink Pasta | Squid Ink Pasta | 'Squid Ink Pasta', 'Black Soba Noodles', 'Stir-fried Black Fungus' | 1.000 | EXACT |
| 533 | Banana | Banana | 'Banana', 'Bananas', 'Banana on a stick' | 1.000 | EXACT |
| 534 | Beef Stew | Cooked Insects | 'Cooked Insects', 'Grilled Insects', 'Fried Insects' | 0.767 | SEMANTIC |
| 535 | Fried Fish | Braised Chicken/ Duck | 'Braised Chicken/ Duck', 'Chicken with herbs and fried onions', 'Braised Quail' | 0.818 | SEMANTIC |
| 536 | Japanese Kaiseki | Japanese meal set | 'Japanese meal set', 'Japanese Kaiseki', 'Korean Meal Set' | 0.826 | SEMANTIC |
| 537 | Oatmeal Chocolate Bar | Oat Protein Bar | 'Oat Protein Bar', 'Oatmeal Chocolate Bar', 'Lan Hua Dou (Peanut Snack)' | 0.856 | SEMANTIC |
| 538 | Sushi | Sushi Salmon | 'Sushi Salmon', 'Salmon Sushi with Miso Soup', 'Sashimi or Shabu-Shabu meat' | 0.878 | SEMANTIC |
| 539 | Mixed Seafood and Steak Platter | Sizzling Meat and Seafood Platter | 'Sizzling Meat and Seafood Platter', 'Seafood and Steak Platter', 'Mixed Seafood and Steak Platter' | 0.903 | SEMANTIC |
| 540 | Cooked White Rice | Rice in Water | 'Rice in Water', 'Noodles or similar grain product', 'Steamed Rice or Grain Dish' | 0.891 | SEMANTIC |
| 541 | Stir-fried greens | Stir-fried Water Spinach | 'Stir-fried Water Spinach', 'Steamed Green Vegetables', 'Stir-fried greens with protein' | 0.880 | SEMANTIC |
| 542 | Chicken Fried Rice | Fried Edible Insects | 'Fried Edible Insects', 'Packaged Chicken Snack', 'Packaged Cooked Chicken' | 0.709 | SEMANTIC |
| 543 | Chocolate-Covered Peanuts | Ferrero Rocher | 'Ferrero Rocher', 'Chocolate Eggs', 'Chocolate-Covered Peanuts' | 0.760 | SEMANTIC |
| 544 | Beef Noodle Soup | Beef Pho | 'Beef Pho', 'Meat Stew with Rice Noodles', 'Hot Beef Noodles' | 0.894 | SEMANTIC |
| 545 | Chicken with Vegetables | Meat Stew with Dried Fruits | 'Meat Stew with Dried Fruits', 'Stir-fried Bamboo Shoots with Vegetables', 'Stir-fried meat with bamboo shoots' | 0.774 | SEMANTIC |
| 546 | Stir-fried liver and vegetables | Stir-fried Liver with Green Peppers | 'Stir-fried Liver with Green Peppers', 'Stir-fried liver with peppers', 'Stir-fried Liver with Peppers' | 0.882 | SEMANTIC |
| 547 | Vegetable platter | Moutai Ice Cream | 'Moutai Ice Cream', 'Foam Topped Beverage', 'Dumplings with drink' | 0.553 | MISS |
| 548 | Mixed Noodles | Instant Noodles with Meat | 'Instant Noodles with Meat', 'Instant Noodles with Beef', 'Instant Noodles with Egg and Meat' | 0.824 | SEMANTIC |
| 549 | Spicy Steamed Fish | Steamed Fish with Spicy Toppings | 'Steamed Fish with Spicy Toppings', 'Steamed Fish with Spicy Topping', 'Steamed Fish with Toppings' | 0.936 | SEMANTIC |
| 550 | Fried Tofu | Fried Tofu Cubes | 'Fried Tofu Cubes', 'Seasoned Croutons', 'Fried Tofu with Herbs' | 0.945 | SEMANTIC |
| 551 | Apricots | Assorted Bread Rolls | 'Assorted Bread Rolls', 'Cute Bread Rolls', 'Pan-fried buns' | 0.679 | MISS |
| 552 | Chopped Lettuce | Boiled Cabbage | 'Boiled Cabbage', 'Cabbage and Carrot Salad', 'Salad with Boiled Eggs' | 0.809 | SEMANTIC |
| 553 | Apple | Raw Apples | 'Raw Apples', 'Apple Snack', 'Apple and Bread Snack' | 0.887 | SEMANTIC |
| 554 | Fried Meatballs | Steamed Meatballs with Dipping Sauce | 'Steamed Meatballs with Dipping Sauce', 'Asian-style meatballs with dipping sauce', 'Fried Balls with Dipping Sauce' | 0.845 | SEMANTIC |
| 555 | Processed Sausage | Fruit-flavored ice pop | 'Fruit-flavored ice pop', 'Processed Meat Stick', 'Strawberry Wafer' | 0.735 | SEMANTIC |
| 556 | Grilled Prawns | Grilled Fish with Side Dishes | 'Grilled Fish with Side Dishes', 'Grilled Fish with Dipping Sauces', 'Grilled Fish and Skewers' | 0.769 | SEMANTIC |
| 557 | Scrambled Eggs with Cassava | Egg Fried Rice with Cassava | 'Egg Fried Rice with Cassava', 'Jollof Rice with Chicken and Eggs', 'Jollof Rice with Chicken and Egg' | 0.893 | SEMANTIC |
| 558 | Spicy Tofu | Fried Tofu in Sauce | 'Fried Tofu in Sauce', 'Spicy Tofu with Sauce', 'Fried tofu with sauce' | 0.899 | SEMANTIC |
| 559 | Pumpkin Soup | Orange Gelatin | 'Orange Gelatin', 'Boiled Pumpkin and Carrots', 'Butternut Squash Soup' | 0.800 | SEMANTIC |
| 560 | Tonkatsu | Pork Cutlet (Tonkatsu) | 'Pork Cutlet (Tonkatsu)', 'Sesame-Crusted Pork', 'Crispy Pork with Cucumber' | 0.897 | SEMANTIC |
| 561 | Walnut | Walnut | 'Walnut', 'Lan Hua Dou (Peanut Snack)', 'Almond Biscuit' | 1.000 | EXACT |
| 562 | Cherries | Red cherries | 'Red cherries', 'Cherries', 'Cherry' | 0.958 | SEMANTIC |
| 563 | Stir-fried Noodles with Beef and Rice | Mixed Meat Dishes with Rice | 'Mixed Meat Dishes with Rice', 'Gyudon with side dishes', 'Gyudon and Udon' | 0.819 | SEMANTIC |
| 564 | Mixed Chinese Dishes | Korean meal with soup and side dishes | 'Korean meal with soup and side dishes', 'Stir-fried greens with pork and side dishes', 'Korean Side Dishes' | 0.857 | SEMANTIC |
| 565 | Noodle Salad | Chajangmyeon | 'Chajangmyeon', 'Soba Noodle Salad', 'Tsukemen (Dipping Noodles)' | 0.684 | MISS |
| 566 | Spicy Dumplings | Spicy Dumplings in Broth | 'Spicy Dumplings in Broth', 'Spicy Soup with Dumplings', 'Dumplings in Spicy Sauce' | 0.931 | SEMANTIC |
| 567 | Fish Soup | Black Sesame Soup | 'Black Sesame Soup', 'Black Sesame Soup with Rice Balls', 'Noodle Soup with Black Sesame Rice' | 0.801 | SEMANTIC |
| 568 | Sushi | Cucumber Sushi Rolls | 'Cucumber Sushi Rolls', 'Vegetable Sushi Roll', 'Vegetable Sushi' | 0.743 | SEMANTIC |
| 569 | Vegetable Stew | Vegetable Salsa | 'Vegetable Salsa', 'Cucumber and Tomato Salad', 'Mixed Vegetable and Meat Dish with Watermelon' | 0.863 | SEMANTIC |
| 570 | Crab | Fried Edible Insects | 'Fried Edible Insects', 'Boiled Crab', 'Whole Crab' | 0.735 | SEMANTIC |
| 571 | Cucumber | Fresh Cucumber | 'Fresh Cucumber', 'Cucumbers', 'Cucumber' | 0.942 | SEMANTIC |
| 572 | Rice Flour | Ginseng Candy | 'Ginseng Candy', 'Korean Red Ginseng Honey Paste', 'Moutai Ice Cream' | 0.753 | SEMANTIC |
| 573 | Steamed Cake | Dorayaki | 'Dorayaki', 'Melon Bread', 'Packaged Rice Ball' | 0.694 | MISS |
| 574 | Green Apples | Sliced Apples and Guavas | 'Sliced Apples and Guavas', 'Green Apples', 'Sliced Green Apples' | 0.779 | SEMANTIC |
| 575 | Meat Soup | Thai Noodle Soup with Grilled Meat | 'Thai Noodle Soup with Grilled Meat', 'Spicy Meat Soup and Stir-Fried Dish', 'Spicy Meat Noodle Soup' | 0.780 | SEMANTIC |
| 576 | Mixed Grilled Skewers | Skewered Meat Platter | 'Skewered Meat Platter', 'Skewered Meat and Fried Snacks', 'Mixed Grilled Meat Platter' | 0.873 | SEMANTIC |
| 577 | Chocolate Frappuccino | Strawberry Cream Frappuccino | 'Strawberry Cream Frappuccino', 'Rose Milkshake', 'Strawberry Milkshake' | 0.801 | SEMANTIC |
| 578 | Macaroons | Macaroons | 'Macaroons', 'Macarons', 'Assorted Packaged Desserts' | 1.000 | EXACT |
| 579 | Hot Pot | Spicy Fish Hot Pot | 'Spicy Fish Hot Pot', 'Tteokbokki and Beef Hot Pot', 'Korean Tteokbokki with assorted side dishes' | 0.885 | SEMANTIC |
| 580 | Chicken Stew | Chicken Stew with Pineapple | 'Chicken Stew with Pineapple', 'Fish Stew with Pineapple', 'Dumplings with Soup and Egg' | 0.868 | SEMANTIC |
| 581 | Seafood Boil | Stir-fried Crab | 'Stir-fried Crab', 'Stir-fried Crabs', 'Crab with Stir-Fried Vegetables' | 0.703 | SEMANTIC |
| 582 | Fish Stew | Garnished Whole Fish | 'Garnished Whole Fish', 'Steamed Fish with Herbs', 'Baked Whole Fish' | 0.801 | SEMANTIC |
| 583 | Grilled Skewers and Fried Snacks | Skewered Meat and Fried Snacks | 'Skewered Meat and Fried Snacks', 'Skewered Meat and Fish Balls', 'Mixed Skewers and Sausages' | 0.928 | SEMANTIC |
| 584 | Grilled Shrimp | Grilled Seafood and Meat Skewers | 'Grilled Seafood and Meat Skewers', 'Grilled Shrimp and Meat Skewers', 'Grilled Prawns' | 0.829 | SEMANTIC |
| 585 | Strawberries | Strawberries | 'Strawberries', 'Fresh Strawberries', 'Sugar-coated Strawberries' | 1.000 | EXACT |
| 586 | Grilled Skewers | Meatballs on a skewer | 'Meatballs on a skewer', 'Lamb Satay', 'Skewered Meat and Fish Balls' | 0.797 | SEMANTIC |
| 587 | Orange | Mandarin Orange | 'Mandarin Orange', 'Orange', 'Orange slice' | 0.837 | SEMANTIC |
| 588 | Fried Chicken Wings | Fried Glazed Dish | 'Fried Glazed Dish', 'Fried Sweet Glazed Bites', 'Sweet Soy Chicken Wings' | 0.816 | SEMANTIC |
| 589 | Fried Chicken | Air Dried Chicken | 'Air Dried Chicken', 'Fried Chicken or Similar Dish', 'Fried Chicken or similar dish' | 0.866 | SEMANTIC |
| 590 | Kung Pao Chicken | Sweet and Sour Pork with Vegetables | 'Sweet and Sour Pork with Vegetables', 'Sweet and Sour Pork', 'Sweet and Sour Pork with Tofu' | 0.781 | SEMANTIC |
| 591 | Chocolate Cheesecake | Chocolate Cake and Dessert Drink | 'Chocolate Cake and Dessert Drink', 'Chocolate Cake and Beverages', 'McDonald's Pie and Soft Drink' | 0.779 | SEMANTIC |
| 592 | Vegetable Rice with Sauce | Yellow Lentil Puree | 'Yellow Lentil Puree', 'Sweet Potato Puree', 'Pounded Yam with Vegetable Sauce' | 0.686 | MISS |
| 593 | Processed Milk and Sausage | Fish-flavored tofu snacks | 'Fish-flavored tofu snacks', 'Seaweed-flavored rice snacks', 'Maoerduo Snack' | 0.696 | MISS |
| 594 | Durian | Peeled Mango | 'Peeled Mango', 'Sliced Mango', 'Packaged Cooked Chicken' | 0.711 | SEMANTIC |
| 595 | Peanuts | Crisp-Coated Peanuts | 'Crisp-Coated Peanuts', 'Roasted Peanuts', 'Seasoned Peanuts' | 0.852 | SEMANTIC |
| 596 | Rolled Omelet | Egg and Ham Tortilla Wraps | 'Egg and Ham Tortilla Wraps', 'Vegetable Omelette Rolls', 'Rolled Omelet with Vegetables' | 0.694 | MISS |
| 597 | Fried Chicken | Fried Chicken Wings and Pizza | 'Fried Chicken Wings and Pizza', 'Fried Chicken and Wings', 'Fried Chicken Wings with Snacks' | 0.843 | SEMANTIC |
| 598 | Grilled Pork Belly | Yakiniku | 'Yakiniku', 'Grilled Meat on Hot Stone', 'Grilled Meat Cubes' | 0.687 | MISS |
| 599 | Steamed Shellfish | Lan Hua Dou (Peanut Snack) | 'Lan Hua Dou (Peanut Snack)', 'Dumplings with Peanuts', 'Flower-shaped dumplings' | 0.702 | SEMANTIC |
| 600 | Crawfish | Crawfish with side dish | 'Crawfish with side dish', 'Crawfish Platter', 'Stuffed Crawfish' | 0.913 | SEMANTIC |
| 601 | Crawfish or Crab Dish | Lobster Paella | 'Lobster Paella', 'Spicy Crab with Corn', 'Seafood Paella' | 0.744 | SEMANTIC |
| 602 | Crawfish | Boiled Crawfish | 'Boiled Crawfish', 'Crawfish with Dipping Sauce', 'Crawfish Feast' | 0.930 | SEMANTIC |
| 603 | Strawberries | Strawberry | 'Strawberry', 'Strawberries', 'Strawberry and Sesame Snack' | 0.950 | SEMANTIC |
| 604 | Braised Chicken Feet | Chicken Feet Dish | 'Chicken Feet Dish', 'Braised Chicken Feet with Lotus Root', 'Chicken Feet with Lemon' | 0.940 | SEMANTIC |
| 605 | Green Peppers | Spicy Green Peppers | 'Spicy Green Peppers', 'Green Peppers', 'Green Chili Peppers' | 0.945 | SEMANTIC |
| 606 | Cream-filled Bun | Cream-filled Cake Roll | 'Cream-filled Cake Roll', 'Cheese Mayonnaise Bread', 'Cream-filled Bread' | 0.877 | SEMANTIC |
| 607 | Banana | Banana | 'Banana', 'Bananas', 'Banana-shaped cake' | 1.000 | EXACT |
| 608 | Stir-fried meat with vegetables | Beef Stir-Fry with Vegetables | 'Beef Stir-Fry with Vegetables', 'Stir-fried Liver with Vegetables', 'Stir-fried liver with vegetables' | 0.901 | SEMANTIC |
| 609 | Milk | Flavored Fermented Milk | 'Flavored Fermented Milk', 'Probiotic Flavored Fermented Milk', 'Flavored Milk' | 0.882 | SEMANTIC |
| 610 | Rice with mixed vegetables | Rice with Fried Snacks and Vegetable Soup | 'Rice with Fried Snacks and Vegetable Soup', 'Mixed Persian Meal', 'Rice with assorted dishes' | 0.862 | SEMANTIC |
| 611 | Noodle with Ground Meat | Noodle dish with green onions | 'Noodle dish with green onions', 'Noodles with Green Onions', 'Stir-fried Noodles with Meat and Lotus Root' | 0.847 | SEMANTIC |
| 612 | Grilled Skewers | Skewered Street Food | 'Skewered Street Food', 'Skewers and Hot Pot', 'Skewered Meat with Noodles' | 0.854 | SEMANTIC |
| 613 | Braised Meat Bowl | Stir-fried Pork Belly with Scallions | 'Stir-fried Pork Belly with Scallions', 'Braised Pork Bowl', 'Stir-fried Pork Belly' | 0.687 | MISS |
| 614 | Fried Pork with Peppers | Sweet Soy Glazed Tofu | 'Sweet Soy Glazed Tofu', 'Sesame Chicken Wings', 'Stir-Fried Chicken Wings' | 0.668 | MISS |
| 615 | Grapes | Boiled Edamame | 'Boiled Edamame', 'Edamame Snack', 'Green Pea Snack' | 0.695 | MISS |
| 616 | Ice Cream Cone | Waffle Cone Ice Cream | 'Waffle Cone Ice Cream', 'Waffle Cone', 'Spicy Soft Serve' | 0.869 | SEMANTIC |
| 617 | Vegetable and Egg Platter | Vegetable platter with dipping sauce | 'Vegetable platter with dipping sauce', 'Mixed Vegetable and Fruit Plate', 'Mixed Vegetable and Protein Plate' | 0.864 | SEMANTIC |
| 618 | Fried Bean Balls | Fried Tofu Balls | 'Fried Tofu Balls', 'Breaded Rice Balls', 'Fried Rice Balls' | 0.889 | SEMANTIC |
| 619 | Fried Chicken Bowl | Mixed Rice Bowl with Fried Chicken | 'Mixed Rice Bowl with Fried Chicken', 'Fried Chicken Bowl', 'Fried Chicken or Similar Dish' | 0.896 | SEMANTIC |
| 620 | Animal-shaped pastries | Assorted Pastries and Drink | 'Assorted Pastries and Drink', 'Assorted Desserts and Beverages', 'Iced Coffee with Lychee' | 0.830 | SEMANTIC |
| 621 | Shrimp and Vegetable Stir-Fry | Shrimp and Snap Peas Stir-Fry | 'Shrimp and Snap Peas Stir-Fry', 'Shrimp with Snow Peas', 'Steamed Shrimp and Vegetables' | 0.811 | SEMANTIC |
| 622 | Fried Chicken Wings | Bungeoppang (Fish-shaped pastry) | 'Bungeoppang (Fish-shaped pastry)', 'Fried Bitter Melon with Pork', 'Fried Sweet Bean Pastry' | 0.642 | MISS |
| 623 | Hawaiian Pizza | Ham Pizza | 'Ham Pizza', 'Shrimp and Ham Pizza', 'Bacon and Corn Pizza' | 0.872 | SEMANTIC |
| 624 | Stir-fried Chicken with Mushrooms | Stir-fried Chicken with Black Fungus | 'Stir-fried Chicken with Black Fungus', 'Scrambled Eggs with Black Fungus', 'Stir-fried Black Fungus' | 0.869 | SEMANTIC |
| 625 | Steamed Dumplings | Steamed Meat Dumplings | 'Steamed Meat Dumplings', 'Momos', 'Traditional Dumplings' | 0.944 | SEMANTIC |
| 626 | Banh Chung | Tamale with mushrooms | 'Tamale with mushrooms', 'Leaf-wrapped rice and cake', 'Tamales' | 0.627 | MISS |
| 627 | Decorated Cake | Cake with Milk | 'Cake with Milk', 'Wedding Cake', 'Zhui Mu Cake' | 0.888 | SEMANTIC |
| 628 | Chicken Pizza | Vegetable Chicken Pizza | 'Vegetable Chicken Pizza', 'Grilled Chicken with Pizza', 'Chicken and Vegetable Pizza' | 0.895 | SEMANTIC |
| 629 | Stir-fried meat with rice | Stir-fried minced meat with rice | 'Stir-fried minced meat with rice', 'Minced Meat with Green Onions', 'Ground Meat with Green Onions' | 0.952 | SEMANTIC |
| 630 | Fermented Bean Paste | Sweetened Bean Paste | 'Sweetened Bean Paste', 'Spicy Shrimp Paste', 'Korean Red Ginseng Honey Paste' | 0.934 | SEMANTIC |
| 631 | Beef slices | Grilled Meat with Citrus | 'Grilled Meat with Citrus', 'Steamed Meat with Ginger', 'Grilled Meat with Herbs' | 0.768 | SEMANTIC |
| 632 | Seafood Stir-fry | Spicy Black Bean Sauce | 'Spicy Black Bean Sauce', 'Seafood in Soy Sauce', 'Spicy Braised Fish' | 0.755 | SEMANTIC |
| 633 | Meat Stew | Spicy Beef Hot Pot | 'Spicy Beef Hot Pot', 'Beef Chili', 'Spicy Ground Meat Dish' | 0.794 | SEMANTIC |
| 634 | Grilled Steak with Vegetables | Mixed Salad with Grilled Steak | 'Mixed Salad with Grilled Steak', 'Grilled Beef with Salad', 'Grilled Ribeye Steak with Vegetables' | 0.802 | SEMANTIC |
| 635 | Bread Rolls | Braided Bread | 'Braided Bread', 'Baked Hot Dog Buns', 'Braided Bread with Coffee' | 0.830 | SEMANTIC |
| 636 | Whole Milk | Calcium-fortified milk | 'Calcium-fortified milk', 'Dairy Protein Drink', 'High-Protein Milk' | 0.896 | SEMANTIC |
| 637 | Milk | Unknown product | 'Unknown product', 'Jelly Strips', 'Maoerduo' | 0.861 | SEMANTIC |
| 638 | Mixed Plate Meal | Vietnamese meal with rice and assorted dishes | 'Vietnamese meal with rice and assorted dishes', 'Mixed Asian Feast', 'Assorted Banquet Dishes' | 0.754 | SEMANTIC |
| 639 | Fried Chicken with Rice | Steamed Fish with Peanuts | 'Steamed Fish with Peanuts', 'Grilled Fish with Sesame', 'Spicy Fish with Peanuts' | 0.712 | SEMANTIC |
| 640 | Pasta with Chicken and Pesto | Spaghetti Aglio e Olio | 'Spaghetti Aglio e Olio', 'Pasta with Pesto and Chicken', 'Pasta with Chicken and Pesto' | 0.772 | SEMANTIC |
| 641 | Grapes | Kyoho Grapes | 'Kyoho Grapes', 'Mixed Grapes', 'Fresh Grapes' | 0.869 | SEMANTIC |
| 642 | Omelet Wrap | Egg Pancake Wrap | 'Egg Pancake Wrap', 'Cheese and Ham Egg Crepe', 'Savory Pancake Wrap' | 0.885 | SEMANTIC |
| 643 | Iced Coffee | Hojicha Latte | 'Hojicha Latte', 'Starbucks Frappuccino and Iced Drink', 'Osmanthus Oolong Milk Tea' | 0.755 | SEMANTIC |
| 644 | Stir-fried Noodles with Beef and Vegetables | Stir-fried vegetables with spaghetti | 'Stir-fried vegetables with spaghetti', 'Vegetable Lo Mein', 'Spicy Noodles with Stir-fried Vegetables' | 0.901 | SEMANTIC |
| 645 | Watermelon | Immature Melon | 'Immature Melon', 'Fruit (likely a type of melon)', 'Fruit (possibly melon)' | 0.793 | SEMANTIC |
| 646 | Seaweed Salad | Salad with Fish Roe | 'Salad with Fish Roe', 'Sashimi Salad', 'Yusheng (Raw Fish Salad)' | 0.715 | SEMANTIC |
| 647 | Chocolate Roll Cake | Chocolate Mint Cheesecake | 'Chocolate Mint Cheesecake', 'Chocolate Cake with Mint Frosting', 'Chocolate Mint Ice Cream Dessert' | 0.686 | MISS |
| 648 | Beef Tacos | Beef Tacos | 'Beef Tacos', 'Tacos with sides', 'Tortillas with meat filling' | 1.000 | EXACT |
| 649 | Braised Chicken | Braised or Glazed Meat Rolls | 'Braised or Glazed Meat Rolls', 'Fried Sesame Balls', 'Spicy Sesame Balls' | 0.830 | SEMANTIC |
| 650 | Fried Spring Rolls | Savory Bread Pudding | 'Savory Bread Pudding', 'Quiche Lorraine', 'Fried Potato Slice' | 0.613 | MISS |
| 651 | Noodle Bowl | Noodles with Bread Roll | 'Noodles with Bread Roll', 'Transparent Noodles', 'Noodles with Fried Items' | 0.815 | SEMANTIC |
| 652 | Fruit Juice | Tapioca Drink | 'Tapioca Drink', 'Mango Bubble Tea', 'Lychee Milk Drink' | 0.838 | SEMANTIC |
| 653 | Spicy Stir-Fried Beef | Stir-fried Liver with Peppers | 'Stir-fried Liver with Peppers', 'Stir-fried liver with peppers', 'Stir-fried Liver with Green Peppers' | 0.884 | SEMANTIC |
| 654 | Crawfish Boil | Spicy Crawfish Platter | 'Spicy Crawfish Platter', 'Spicy Crawfish and Side Dishes', 'Crawfish Platter' | 0.799 | SEMANTIC |
| 655 | Braised Pork with Abalone | Grilled Pineapple and Skewered Meat | 'Grilled Pineapple and Skewered Meat', 'Char Siu (Chinese BBQ Pork)', 'Chinese BBQ Pork (Char Siu)' | 0.647 | MISS |
| 656 | Fried Egg Noodles | Fried Fish Noodle Soup | 'Fried Fish Noodle Soup', 'Fried Noodles with Egg and Steamed Bun', 'Chinese pancake with fillings and cold noodles' | 0.836 | SEMANTIC |
| 657 | Vegetable Noodle Soup | Oyster Noodle Soup | 'Oyster Noodle Soup', 'Fish Noodle Soup', 'Egg Noodle Soup' | 0.906 | SEMANTIC |
| 658 | Grilled Meat Rolls | Fried Snack on a Stick | 'Fried Snack on a Stick', 'Fried Sausage on a Stick', 'Grilled Cheese Skewers' | 0.723 | SEMANTIC |
| 659 | Stir-fried Green Peppers with Meat | Stir-fried Pork with Green Beans | 'Stir-fried Pork with Green Beans', 'Pork and Green Beans Stir-fry', 'Stir-fried Green Beans with Pork' | 0.804 | SEMANTIC |
| 660 | Vegetable Noodle Salad | Stir-fried vegetables with spaghetti | 'Stir-fried vegetables with spaghetti', 'Stir-fried Glass Noodles with Vegetables', 'Fried Noodles with Vegetables' | 0.860 | SEMANTIC |
| 661 | Grilled Meat | Grilled Pork with Onions | 'Grilled Pork with Onions', 'Grilled Flavor Pork', 'Sliced Pork with Garlic Sauce' | 0.833 | SEMANTIC |
| 662 | Fried Skewers and Potatoes | Fried Sausage on a Stick | 'Fried Sausage on a Stick', 'Fried Snack on a Stick', 'Mixed Asian Street Food' | 0.799 | SEMANTIC |
| 663 | Coffee Snack | Chocolate-flavored drink | 'Chocolate-flavored drink', 'Cocoa Wafers', 'Coffee Candy' | 0.865 | SEMANTIC |
| 664 | Scrambled Eggs with Tomatoes | Scrambled Eggs with Ham and Tomatoes | 'Scrambled Eggs with Ham and Tomatoes', 'Scrambled Eggs with Tomatoes and Cheese', 'Scrambled Eggs with Tomatoes' | 0.946 | SEMANTIC |
| 665 | Spicy Crawfish | Vegetable Japchae | 'Vegetable Japchae', 'Japchae', 'Stir-fried Crab' | 0.588 | MISS |
| 666 | Decorative Rice Dish | Bear-shaped rice with meat and vegetables | 'Bear-shaped rice with meat and vegetables', 'Bear-shaped rice dish with pumpkin', 'Bear-shaped rice with curry and fried vegetables' | 0.709 | SEMANTIC |
| 667 | Fruit Juice | Milo drink | 'Milo drink', 'Green Apple Flavored Milk', 'Soya Milk' | 0.806 | SEMANTIC |
| 668 | Dumplings | Siu Mai | 'Siu Mai', 'Mixed Plate with Steamed Dumplings', 'Steamed Dumplings with Sauce' | 0.706 | SEMANTIC |
| 669 | Sushi platter | Seaweed Rice Bowl with Fried Fish | 'Seaweed Rice Bowl with Fried Fish', 'Salad with Fish Roe', 'Yusheng (Raw Fish Salad)' | 0.657 | MISS |
| 670 | Vegetable Salad | Steamed Rice Cake with Vegetables | 'Steamed Rice Cake with Vegetables', 'Stir-fried Peas with Meat', 'Stir-fried Peas' | 0.804 | SEMANTIC |
| 671 | Durian | Mango Sticky Rice Dumpling | 'Mango Sticky Rice Dumpling', 'Stuffed Passion Fruit', 'Sweet Yellow Dumplings' | 0.691 | MISS |
| 672 | Mixed Rice with Meat and Dumplings | Mixed Buffet Dishes | 'Mixed Buffet Dishes', 'Mixed Asian Meal with Pastries', 'Assorted Banquet Dishes' | 0.758 | SEMANTIC |
| 673 | Mapo Tofu | Fried Tofu in Tomato Sauce | 'Fried Tofu in Tomato Sauce', 'Spicy Tofu and Fish Stew', 'Stir-fried Tofu with Potatoes' | 0.728 | SEMANTIC |
| 674 | Meat Salad | Pad Thai | 'Pad Thai', 'Thai style pork rice', 'Vietnamese Pork with Rice Noodles' | 0.708 | SEMANTIC |
| 675 | Savory Crispy Cups | Flower-shaped dumplings | 'Flower-shaped dumplings', 'Spiced Lotus Root Salad', 'Flower-shaped pastries with dessert' | 0.677 | MISS |
| 676 | Crinkle-Cut Fries | Fried Cassava Fries | 'Fried Cassava Fries', 'Fried Corn Snack', 'Steamed Baby Corn with Sauce' | 0.753 | SEMANTIC |
| 677 | Sweet and Sour Meatballs | Spicy Sesame Balls | 'Spicy Sesame Balls', 'Fried Sesame Balls', 'Fried Bean Balls' | 0.798 | SEMANTIC |
| 678 | Roasted Chicken | Steamed Chicken with Pumpkin | 'Steamed Chicken with Pumpkin', 'Boiled Chicken Parts', 'Sliced Cooked Chicken' | 0.811 | SEMANTIC |
| 679 | Steamed Bun | Pig-shaped steamed buns | 'Pig-shaped steamed buns', 'Xue Yan (Snow Fungus)', 'Steamed Pig Buns' | 0.827 | SEMANTIC |
| 680 | Roasted Duck with Rice and Vegetables | Braised Pork with Soybeans | 'Braised Pork with Soybeans', 'Steamed Pork Ribs', 'Spicy Braised Pork Ribs' | 0.686 | MISS |
| 681 | Steak with Pasta | Steak and Pasta Plate | 'Steak and Pasta Plate', 'Noodle and Beef Plate', 'Shrimp Pasta and Grilled Steak' | 0.942 | SEMANTIC |
| 682 | Sashimi Platter | Sashimi Platter with Rice | 'Sashimi Platter with Rice', 'Sashimi or Raw Seafood Platter', 'Sashimi with Fish' | 0.940 | SEMANTIC |
| 683 | Fried Chicken Meal | KFC Chicken Meal | 'KFC Chicken Meal', 'KFC Fried Chicken Meal', 'McDonald's Chicken Nuggets' | 0.926 | SEMANTIC |
| 684 | Stir-fried Greens with Mushrooms | Stir-fried Mushrooms and Greens | 'Stir-fried Mushrooms and Greens', 'Stir-fried mushrooms and greens', 'Stir-fried Greens with Mushrooms' | 0.964 | SEMANTIC |
| 685 | Vegetable Sandwich | Bánh Mì | 'Bánh Mì', 'Bánh mì', 'Cabbage and Carrot Salad' | 0.688 | MISS |
| 686 | Cooked Beef | Cooked Steak | 'Cooked Steak', 'Beef Tenderloin', 'Sliced Steak' | 0.930 | SEMANTIC |
| 687 | Nectarine | Nectarine | 'Nectarine', 'Nectarines', 'Raw Mango' | 1.000 | EXACT |
| 688 | Vegetable Wrap | Savory Pancake Wrap | 'Savory Pancake Wrap', 'Mixed Wraps with Fries', 'Vegetable Tortilla Wrap' | 0.868 | SEMANTIC |
| 689 | Spicy Noodle Soup | Tofu Soup with Fried Bread | 'Tofu Soup with Fried Bread', 'Fish Maw Soup', 'Noodle Soup with Fried Tofu' | 0.683 | MISS |
| 690 | Oysters on the half shell | Steamed Oysters | 'Steamed Oysters', 'Steamed Oysters with Glass Noodles', 'Spicy Oysters' | 0.899 | SEMANTIC |
| 691 | Spaghetti | Enoki Mushroom Noodles | 'Enoki Mushroom Noodles', 'Enoki Mushrooms', 'Shredded Noodles' | 0.754 | SEMANTIC |
| 692 | Hamburger | Chinese Burger | 'Chinese Burger', 'Chinese Rice Burger', 'Cheeseburger with Soft Drink' | 0.899 | SEMANTIC |
| 693 | Baked Pastry | Sliced Mushrooms | 'Sliced Mushrooms', 'Jalebi', 'Uncooked Pasta' | 0.725 | SEMANTIC |
| 694 | Beef Chow Mein | Stir-fried Noodles with Green Beans | 'Stir-fried Noodles with Green Beans', 'Stir-fried Green Beans with Black Fungus', 'Stir-fried Beef with Green Beans' | 0.735 | SEMANTIC |
| 695 | Grilled Meat Skewers | Grilled Meat Skewers with Salad | 'Grilled Meat Skewers with Salad', 'Grilled Meat Skewers with Vegetables', 'Grilled Meat Skewers with Stir-fried Vegetables' | 0.898 | SEMANTIC |
| 696 | Vegetable Terrine | Cucumber Sauce | 'Cucumber Sauce', 'Fish in Green Sauce', 'Vegetable Puree' | 0.696 | MISS |
| 697 | Noodle Soup with Fish | Noodles with Tofu and Soup | 'Noodles with Tofu and Soup', 'Green Noodle Soup', 'Noodle Soup with Seaweed' | 0.851 | SEMANTIC |
| 698 | Shredded Vegetables | Sliced Daikon Radish | 'Sliced Daikon Radish', 'Raw Daikon Salad', 'Stir-fried Cabbage and Bean Sprouts' | 0.677 | MISS |
| 699 | Fried Pork with Vegetables | Mixed Filipino Dishes | 'Mixed Filipino Dishes', 'Chicharrón', 'Ikan Bolu Goreng Sambel' | 0.745 | SEMANTIC |
| 700 | Vegetable Stir-Fry | Stir-fried Bamboo Shoots with Vegetables | 'Stir-fried Bamboo Shoots with Vegetables', 'Stir-fried Bamboo Shoots with Meat', 'Stir-fried Bamboo Shoots' | 0.828 | SEMANTIC |
| 701 | Mixed Plate Meal | Mixed Eastern European Cuisine | 'Mixed Eastern European Cuisine', 'Baked Dish with Beverage', 'Potato and Salad Plate' | 0.866 | SEMANTIC |
| 702 | Chicken Feet with Peanuts | Braised Pork with Beans | 'Braised Pork with Beans', 'Mixed Meat and Beans Bowl', 'Grain and Bean Casserole' | 0.731 | SEMANTIC |
| 703 | Spicy Black Noodles | Stir-fried Insects | 'Stir-fried Insects', 'Stir-fried insects', 'Stir-fried Octopus' | 0.789 | SEMANTIC |
| 704 | Stir-fried Chicken with Mushrooms and Tomato Scrambled Eggs | Fried Rice and Stir-Fried Noodles with Roasted Meat | 'Fried Rice and Stir-Fried Noodles with Roasted Meat', 'Stir-fried meat and tofu dishes', 'Stir-fried beef with vegetables, scrambled eggs, and rice' | 0.713 | SEMANTIC |
| 705 | Shrimp Stir-Fry | Noodle Salad with Shrimp | 'Noodle Salad with Shrimp', 'Shrimp with Bean Sprouts', 'Shrimp Soba Noodle Salad' | 0.833 | SEMANTIC |
| 706 | Spicy Chicken Dish | Roast Duck and Char Siu Rice | 'Roast Duck and Char Siu Rice', 'Sesame-Crusted Pork', 'Beijing Roast Duck' | 0.733 | SEMANTIC |
| 707 | Spicy Noodles | Spicy Sichuan Cuisine | 'Spicy Sichuan Cuisine', 'Shanxi-style Stir-fried Meat', 'Paella' | 0.839 | SEMANTIC |
| 708 | Noodle Soup | Noodle with Beef | 'Noodle with Beef', 'Hong Kong Style Noodles', 'Jiaomijiao Pickled Pepper Beef Noodle' | 0.900 | SEMANTIC |
| 709 | Mixed Salad | Mango Salad | 'Mango Salad', 'Salad with Mango', 'Shredded Mango Salad' | 0.740 | SEMANTIC |
| 710 | Chicken Sandwich | Breakfast Sandwich with Coffee | 'Breakfast Sandwich with Coffee', 'Crispy Chicken Muffin Meal', 'Sausage McGriddle' | 0.859 | SEMANTIC |
| 711 | Dumplings | Steamed dumplings with sauce | 'Steamed dumplings with sauce', 'Steamed Dumplings with Sauce', 'Dumplings in Syrup' | 0.878 | SEMANTIC |
| 712 | Flatbread | Tortillas with meat filling | 'Tortillas with meat filling', 'Stuffed Tortillas', 'Quesadilla' | 0.772 | SEMANTIC |
| 713 | Stir-fried Meat with Green Peppers | Stir-fried Peppers with Meat | 'Stir-fried Peppers with Meat', 'Stir-fried Snap Peas with Meat', 'Stir-fried Chicken Feet with Green Beans' | 0.900 | SEMANTIC |
| 714 | Watermelon | Cut Watermelon | 'Cut Watermelon', 'Sliced Watermelon', 'Watermelon' | 0.919 | SEMANTIC |
| 715 | Seafood Salad | Mixed Vegetable and Fruit Meal | 'Mixed Vegetable and Fruit Meal', 'Mixed Vegetable and Tofu Dishes', 'Mixed Vegetable Dishes with Rice' | 0.751 | SEMANTIC |
| 716 | Scrambled Eggs with Tomatoes | Spicy Chicken and Tomato Egg Stir-fry | 'Spicy Chicken and Tomato Egg Stir-fry', 'Spicy Tofu Dish', 'Tomato and Egg Stir-fry with Chicken' | 0.775 | SEMANTIC |
| 717 | Hot Pot | Korean Hot Pot | 'Korean Hot Pot', 'Vietnamese Hot Pot', 'Barbecue or Hot Pot Ingredients' | 0.910 | SEMANTIC |
| 718 | Shrimp in Cream Sauce with Rice | Shrimp in Cream Sauce with Rice | 'Shrimp in Cream Sauce with Rice', 'Shrimp in Cream Sauce', 'Creamy Chicken and Shrimp with Rice' | 1.000 | EXACT |
| 719 | Braised Pork | Braised Pork Hock | 'Braised Pork Hock', 'Spicy Braised Duck', 'Braised Duck with Rice' | 0.955 | SEMANTIC |
| 720 | Red Bean Pastry | Red Bean Pastry | 'Red Bean Pastry', 'Croissant with Red Bean Paste and Butter', 'Red Bean Bread' | 1.000 | EXACT |
| 721 | Vegetable Stir-Fry | Pork and Bamboo Shoot Stew | 'Pork and Bamboo Shoot Stew', 'Braised Meat with Fried Tofu', 'Braised Daikon with Pork' | 0.751 | SEMANTIC |
| 722 | Chili Crab | Abalone in Spicy Sauce | 'Abalone in Spicy Sauce', 'Chili Crab', 'Steamed Fish with Spicy Sauce' | 0.699 | MISS |
| 723 | Nutritious Breakfast Plate | Steamed Meat and Egg Dish | 'Steamed Meat and Egg Dish', 'Boiled Egg and Wrapped Food', 'Sliced Pork and Soft-Boiled Eggs' | 0.739 | SEMANTIC |
| 724 | Orange Juice | Orange Drink | 'Orange Drink', 'Orange and Orange Juice', 'Orange Tea' | 0.917 | SEMANTIC |
| 725 | Spicy Noodle Soup | Tteokbokki with fried snacks | 'Tteokbokki with fried snacks', 'Tteokbokki with Noodles', 'Tteokbokki with Cheese' | 0.661 | MISS |
| 726 | Noodle Soup with Rice | Korean Stew (e.g., Jjigae) | 'Korean Stew (e.g., Jjigae)', 'Korean meal with soup and rice', 'Korean meal with rice and soup' | 0.680 | MISS |
| 727 | Beef Noodle Soup | Beef with Pasta and Vegetables | 'Beef with Pasta and Vegetables', 'Stir-fried noodles with beef and vegetables', 'Stir-fried Noodles with Beef and Vegetables' | 0.787 | SEMANTIC |
| 728 | Steamed Broccoli with Almonds | Steamed Fish with Broccoli | 'Steamed Fish with Broccoli', 'Steamed Fish with Broccoli and Rice', 'Herb Steamed Fish' | 0.812 | SEMANTIC |
| 729 | Fried Dumplings | Sweet Rice Dumplings | 'Sweet Rice Dumplings', 'Rice Dumplings', 'Sweet Rice Balls in Ginger Syrup' | 0.869 | SEMANTIC |
| 730 | Fried Chicken Drumsticks | Cooked Chicken Drumstick | 'Cooked Chicken Drumstick', 'Ayam Palekko', 'Cooked Chicken Drumsticks' | 0.873 | SEMANTIC |
| 731 | Fried Rice with Chestnuts and Vegetables | Braised Rice with Meat and Vegetables | 'Braised Rice with Meat and Vegetables', 'Stir-fried Rice with Sausage', 'Stir-fried rice with sausage' | 0.850 | SEMANTIC |
| 732 | Fried Dough with Ice Cream | Pastry with Ice Cream | 'Pastry with Ice Cream', 'Pancakes with Ice Cream and Mango', 'Fluffy Pancakes with Mont Blanc' | 0.845 | SEMANTIC |
| 733 | Beef and Potato Stew | Sweet and Sour Pork with Rice | 'Sweet and Sour Pork with Rice', 'Sweet and Sour Pork with Tofu', 'Sweet and Sour Pork with Vegetables' | 0.788 | SEMANTIC |
| 734 | Strawberries | Strawberries | 'Strawberries', 'Fresh Strawberries', 'Strawberry' | 1.000 | EXACT |
| 735 | Boiled Chicken | Steamed Chicken Feet | 'Steamed Chicken Feet', 'Bungeoppang (Fish-shaped pastry)', 'Braised Chicken Feet with Lotus Root' | 0.789 | SEMANTIC |
| 736 | Assorted Beef Cuts | Grilled meat with vegetables and raw beef tartare | 'Grilled meat with vegetables and raw beef tartare', 'Grilled Meat on Hot Stone', 'Assorted Wagyu Beef' | 0.728 | SEMANTIC |
| 737 | Stir-fried Noodles | Pasta with Fried Onions | 'Pasta with Fried Onions', 'Stir-fried noodles with meat', 'Stir-fried Noodles with Meat' | 0.838 | SEMANTIC |
| 738 | Cheese Pizza | Cheese Pizza | 'Cheese Pizza', 'Cheese and Pepperoni Pizza', 'Cheesy Pizza' | 1.000 | EXACT |
| 739 | Fried Spring Rolls | Fried Tofu with Meat Filling | 'Fried Tofu with Meat Filling', 'Cassava with Egg Sauce', 'Fried Tofu or Similar' | 0.800 | SEMANTIC |
| 740 | Noodles | Lotus Root and Edamame Stir-fry | 'Lotus Root and Edamame Stir-fry', 'Lotus Root Snack', 'Stir-fried Lotus Root and Noodles' | 0.587 | MISS |
| 741 | Stir-fried Cabbage with Eggs | Chicken Noodles with Egg | 'Chicken Noodles with Egg', 'Cheesy Chicken Noodles', 'Fried Egg Noodles' | 0.771 | SEMANTIC |
| 742 | Yellow Plums and Apple | Yellow fruits (e.g., plums or similar) | 'Yellow fruits (e.g., plums or similar)', 'Yellow cherries', 'Indian Gooseberries' | 0.845 | SEMANTIC |
| 743 | Cashew Fruit | Peeled Mango | 'Peeled Mango', 'Frozen Mango', 'Cashew Fruit' | 0.716 | SEMANTIC |
| 744 | Salmon fillets | Salmon fillets | 'Salmon fillets', 'Salmon fillet', 'Salmon Fillet' | 1.000 | EXACT |
| 745 | Roti | Cheese Paratha | 'Cheese Paratha', 'Paratha', 'Stuffed Paratha with Yogurt' | 0.774 | SEMANTIC |
| 746 | Mixed Rice Bowl | Stir-fried beef with vegetables, scrambled eggs, and rice | 'Stir-fried beef with vegetables, scrambled eggs, and rice', 'Rice with Grilled Meat and Vegetables', 'Mixed rice bowl with curry and vegetables' | 0.712 | SEMANTIC |
| 747 | Vegetable Stew | Curry with Potatoes and Meat | 'Curry with Potatoes and Meat', 'Meat and Potato Curry', 'Meat Curry with Potatoes' | 0.788 | SEMANTIC |
| 748 | Spicy Black Rice Bowl | Spicy Black Rice Bowl | 'Spicy Black Rice Bowl', 'Stir-fried Seaweed and Vegetables', 'Stir-fried Seaweed' | 1.000 | EXACT |
| 749 | Apple | Nectarine | 'Nectarine', 'Apple', 'Apple Snack' | 0.867 | SEMANTIC |
| 750 | Milk Tea | Hojicha Latte | 'Hojicha Latte', 'Frappuccino', 'Tapioca Drink' | 0.788 | SEMANTIC |
| 751 | Assorted Chinese Dishes | Korean side dishes (Banchan) | 'Korean side dishes (Banchan)', 'Korean mixed dishes', 'Korean Side Dishes' | 0.752 | SEMANTIC |
| 752 | Pizza | Stuffed Spaghetti Squash | 'Stuffed Spaghetti Squash', 'Savory Rice Cake with Meat', 'Savory Rice Cake' | 0.557 | MISS |
| 753 | Stir-fried Noodles with Vegetables | Spicy Chicken with Sichuan Peppercorns | 'Spicy Chicken with Sichuan Peppercorns', 'Spicy Peanut Stir-fry', 'Spicy Chicken with Sesame' | 0.681 | MISS |
| 754 | Lotus Seeds | Wasabi Peas | 'Wasabi Peas', 'Green Pea Snack', 'Canned Green Peas' | 0.692 | MISS |
| 755 | Spicy Shrimp Stir-Fry | Spicy Sichuan Cuisine | 'Spicy Sichuan Cuisine', 'Spicy Stir-Fried Squid', 'Stir-fried Squid with Peppers' | 0.730 | SEMANTIC |
| 756 | Mixed Berries | Bowl of Strawberries | 'Bowl of Strawberries', 'Fresh Strawberries', 'Frozen Strawberries' | 0.789 | SEMANTIC |
| 757 | Bánh Xèo | Vietnamese Omelet (Bánh Xèo) | 'Vietnamese Omelet (Bánh Xèo)', 'Vietnamese Omelette', 'Vietnamese Pancake (Bánh Xèo)' | 0.787 | SEMANTIC |
| 758 | Milk Sachima | Sea Salt Soda Crackers | 'Sea Salt Soda Crackers', 'Packaged Chicken Snack', 'Packaged Sausage Snack' | 0.706 | SEMANTIC |
| 759 | French Fries | Fried Cassava Fries | 'Fried Cassava Fries', 'Seasoned Fries', 'Crinkle-Cut Fries' | 0.814 | SEMANTIC |
| 760 | Watermelon | Fried Rice with Watermelon | 'Fried Rice with Watermelon', 'Watermelon and Melon Salad', 'Cut Watermelon' | 0.774 | SEMANTIC |
| 761 | Potato Chips | Sizzled Barbecue Flavor Chips | 'Sizzled Barbecue Flavor Chips', 'Fried Cornmeal Triangles', 'Baked Potato Chips' | 0.857 | SEMANTIC |
| 762 | Fried snacks with sauces | Fried Fish with Chips | 'Fried Fish with Chips', 'Fried Fish and Chips', 'Fried Fish with Creamy Sauce' | 0.831 | SEMANTIC |
| 763 | Shrimp and Cheeseburger | Assorted Burgers and Pastries | 'Assorted Burgers and Pastries', 'Colorful Burgers', 'Cheese Burgers' | 0.758 | SEMANTIC |
| 764 | Grilled Pork with Rice | Grilled Meat with Rice and Egg | 'Grilled Meat with Rice and Egg', 'Pork Cutlet Rice with Egg', 'Fried Pork with Egg and Rice' | 0.895 | SEMANTIC |
| 765 | Mixed Meat and Vegetable Platter | Stir-fried greens with meat and side dishes | 'Stir-fried greens with meat and side dishes', 'Stir-fried greens with pork and side dishes', 'Hot Pot and Side Dishes' | 0.707 | SEMANTIC |
| 766 | Assorted Snacks | Instant Noodles and Snacks | 'Instant Noodles and Snacks', 'Assorted Sweets and Snacks', 'Assorted Bakery Goods' | 0.858 | SEMANTIC |
| 767 | Baked Corn Muffins | Baked Cornmeal Cakes | 'Baked Cornmeal Cakes', 'Fried Cornmeal Cakes', 'Baked Cornmeal Cakes with Yogurt' | 0.831 | SEMANTIC |
| 768 | Spicy Noodle Dish | Korean meal with rice and kimchi | 'Korean meal with rice and kimchi', 'Mixed Korean Meal', 'Korean Stew Meal' | 0.686 | MISS |
| 769 | Pesto Pasta and Burger Meal | Pesto Pasta and Burger Meal | 'Pesto Pasta and Burger Meal', 'Cucumber and Chickpea Salad', 'Noodle Salad with Peanuts' | 1.000 | EXACT |
| 770 | Noodle Soup with Tofu | Bún riêu | 'Bún riêu', 'Vietnamese Meatball Soup', 'Vietnamese meatball soup with noodles' | 0.688 | MISS |
| 771 | Braised Pork Belly | Braised Pork Belly | 'Braised Pork Belly', 'Stir-fried Pork Belly with Scallions', 'Steamed Pork Belly' | 1.000 | EXACT |
| 772 | Mixed Asian Cuisine | Mixed Vegetable Plate with Egg and Soup | 'Mixed Vegetable Plate with Egg and Soup', 'Mixed Vegetable and Tofu Dishes', 'Spicy Mixed Dishes with Desserts' | 0.680 | MISS |
| 773 | Beef Soup | Vietnamese Meatball Soup | 'Vietnamese Meatball Soup', 'Meatball Soup with Greens', 'Beef Meatball Soup' | 0.778 | SEMANTIC |
| 774 | Rasgulla | Tangyuan | 'Tangyuan', 'Tangyuan (Glutinous Rice Balls)', 'Tangyuan (glutinous rice balls)' | 0.762 | SEMANTIC |
| 775 | Prosciutto Salad | Mixed Salad with Prosciutto | 'Mixed Salad with Prosciutto', 'Salad with Prosciutto and Figs', 'Salad with Prosciutto and Fruits' | 0.934 | SEMANTIC |
| 776 | Pasta with Plantains | Meatball Fries | 'Meatball Fries', 'Fries with Sausage', 'Mixed Plate with Fries and Meat' | 0.663 | MISS |
| 777 | Steamed Fish | Salmon Sashimi with Roe | 'Salmon Sashimi with Roe', 'Salmon Sashimi', 'Steamed Fish with Toppings' | 0.710 | SEMANTIC |
| 778 | Sliced Pork and Soft-Boiled Eggs | Sliced Pork and Soft-Boiled Eggs | 'Sliced Pork and Soft-Boiled Eggs', 'Pork Slices with Egg', 'Sliced Chicken with Egg' | 1.000 | EXACT |
| 779 | Oyster Sauce | Tomato Ale | 'Tomato Ale', 'Mung Bean Craft Beer', 'Tsingtao Beer' | 0.715 | SEMANTIC |
| 780 | Crawfish Noodles | Stir-fried Noodles with Chicken and Peppers | 'Stir-fried Noodles with Chicken and Peppers', 'Spicy Noodles with Stir-fried Vegetables', 'Spicy Noodles and Stir-Fried Vegetables' | 0.783 | SEMANTIC |
| 781 | Steamed Shrimp | Shrimp with Yellow Topping | 'Shrimp with Yellow Topping', 'Steamed Shrimp with Dipping Sauce', 'Shrimp or similar seafood dish' | 0.827 | SEMANTIC |
| 782 | Stir-fried Tofu with Vegetables | Stir-fried Tofu with Edamame | 'Stir-fried Tofu with Edamame', 'Stir-fried Tofu and Edamame', 'Steamed Fish with Tofu and Vegetables' | 0.841 | SEMANTIC |
| 783 | Sunflower Seeds | Sunflower Seeds | 'Sunflower Seeds', 'Roasted Sunflower Seeds', 'Canned Toddy Palm Seeds' | 1.000 | EXACT |
| 784 | Sweet and Sour Pork | Sweet and Sour Pork with Vegetables | 'Sweet and Sour Pork with Vegetables', 'Sweet and Sour Pork', 'Stir-fried Apples with Pork' | 0.956 | SEMANTIC |
| 785 | Meat Soup | Noodle Soup with Century Egg | 'Noodle Soup with Century Egg', 'Mung Bean Soup with Glutinous Rice Balls', 'Century Egg Congee' | 0.675 | MISS |
| 786 | Mixed Asian Dishes | Mixed Chinese Seafood and Meat Dishes | 'Mixed Chinese Seafood and Meat Dishes', 'Shrimp and mixed dishes', 'Shrimp and Mixed Dishes' | 0.903 | SEMANTIC |
| 787 | Roasted Duck | Roast Duck and Pork Platter | 'Roast Duck and Pork Platter', 'Roast Duck with Sauce', 'Roast Duck and Pork' | 0.867 | SEMANTIC |
| 788 | Vegetable Tofu Soup | Tofu Seaweed Soup | 'Tofu Seaweed Soup', 'Tteokguk (Rice Cake Soup)', 'Radish Soup' | 0.888 | SEMANTIC |
| 789 | Seafood Congee | Seafood Congee | 'Seafood Congee', 'Seafood Rice Porridge', 'Seafood Porridge' | 1.000 | EXACT |
| 790 | Steamed Buns | Steamed buns with sliced meat | 'Steamed buns with sliced meat', 'Pig-shaped steamed buns', 'Steamed Buns and Bread' | 0.864 | SEMANTIC |
| 791 | Mixed Nuts or Snack Mix | Lan Hua Dou (Peanut Snack) | 'Lan Hua Dou (Peanut Snack)', 'Dried Bean Curd', 'Spicy Dried Fish Snack' | 0.698 | MISS |
| 792 | Steamed Buns | Steamed Buns with Meat | 'Steamed Buns with Meat', 'Steamed buns with sliced meat', 'Steamed Buns and Dumplings' | 0.928 | SEMANTIC |
| 793 | Mixed Asian Meal | Spicy Noodle Soup and Fried Chicken | 'Spicy Noodle Soup and Fried Chicken', 'Mixed Seafood and Noodle Dishes', 'Seafood Noodle Soup and Fried Items' | 0.739 | SEMANTIC |
| 794 | Assorted Asian Dishes | Dumplings with noodles and side dish | 'Dumplings with noodles and side dish', 'Korean BBQ side dishes', 'Korean Side Dishes' | 0.791 | SEMANTIC |
| 795 | Roasted Chicken with Vegetables | Roasted Turkey with Vegetables | 'Roasted Turkey with Vegetables', 'Roast Chicken with Sides', 'Roasted Turkey with Sides' | 0.964 | SEMANTIC |
| 796 | Mixed Meat Platter | Mixed Meat and Tofu Platter | 'Mixed Meat and Tofu Platter', 'Seafood and Tofu Platter', 'Mixed Chinese Seafood and Meat Platter' | 0.918 | SEMANTIC |
| 797 | Katsu Curry | Curry with Tonkatsu | 'Curry with Tonkatsu', 'Tonkatsu with Hamburger Steak', 'Curry Rice with Fried Chicken' | 0.903 | SEMANTIC |
| 798 | Chocolate Bean Soup | Chocolate Beverage | 'Chocolate Beverage', 'Chocolate-flavored drink', 'Chocolate Drink' | 0.827 | SEMANTIC |
| 799 | Seafood Stir-fry | Takeout meal with drink | 'Takeout meal with drink', 'Lobster Tofu Stir-fry', 'Lobster Tofu Stir Fry' | 0.696 | MISS |
| 800 | Fried Chicken Wings | Baked Chicken Wings | 'Baked Chicken Wings', 'BBQ Chicken Drumsticks', 'Glazed Chicken Wings' | 0.930 | SEMANTIC |
| 801 | Rice with sauce and egg | Rice with Fried Plantains and Tomato Sauce | 'Rice with Fried Plantains and Tomato Sauce', 'Nasi Kuning', 'Nasi Kuning with Chicken' | 0.760 | SEMANTIC |
| 802 | Braised Meat | Pork Adobo | 'Pork Adobo', 'Pork and Celery Stir-fry', 'Stir-fried Chicken Liver' | 0.785 | SEMANTIC |
| 803 | Rice Porridge | Sweet Rice Porridge | 'Sweet Rice Porridge', 'Rice Porridge', 'Coconut Sago Dessert' | 0.985 | SEMANTIC |
| 804 | Spaghetti Bolognese | Pasta with Tomato Sauce and Shrimp | 'Pasta with Tomato Sauce and Shrimp', 'Pasta with Sausages and Tomatoes', 'Spaghetti with Shellfish' | 0.744 | SEMANTIC |
| 805 | Cooked Chicken Wings | Boneless Pickled Duck Feet | 'Boneless Pickled Duck Feet', 'Smoked Chicken Feet', 'Spicy Duck Feet' | 0.802 | SEMANTIC |
| 806 | Sesame Chicken | Chicken with Corn and Sweet Potatoes | 'Chicken with Corn and Sweet Potatoes', 'Stir-fried bamboo shoots with pork', 'Stir-fried Bamboo Shoots with Pork' | 0.688 | MISS |
| 807 | Noodle Soup with Meatballs and Fish | Bún chả | 'Bún chả', 'Steamed Meat and Mushroom Dish', 'Mixed Fish Ball Soup' | 0.649 | MISS |
| 808 | Shrimp and Egg Pizza | Fried Shrimp with Toppings | 'Fried Shrimp with Toppings', 'Grilled Shrimp Platter', 'Grilled Shrimp and Meat Platter' | 0.733 | SEMANTIC |
| 809 | Chicken Salad | Chicken and Green Peppers Stir-Fry | 'Chicken and Green Peppers Stir-Fry', 'Spicy Chicken Stir-Fry', 'Spicy Chicken Stir-fry' | 0.680 | MISS |
| 810 | Roasted Chicken | Seafood Platter with Omelet | 'Seafood Platter with Omelet', 'Steamed Crabs with Side Dishes', 'Crab and assorted dishes' | 0.686 | MISS |
| 811 | Heart-shaped pancake | Heart-shaped pancake | 'Heart-shaped pancake', 'Hokkaido Flavored Round Cake', 'Tahu Telor' | 1.000 | EXACT |
| 812 | Noodle Soup | Tsukemen (Dipping Noodles) | 'Tsukemen (Dipping Noodles)', 'Noodle Soup with Egg', 'Noodle dish with boiled eggs' | 0.670 | MISS |
| 813 | Fried Snack Platter | Crispy Chicken Bites | 'Crispy Chicken Bites', 'Mixed Fried Snacks', 'Mixed Fried Platter' | 0.807 | SEMANTIC |
| 814 | Lentil Curry with Rice | Dal with Rice | 'Dal with Rice', 'Rice with Eggs and Pumpkin', 'Rice with Lentil Curry' | 0.834 | SEMANTIC |
| 815 | Fruit Platter | Fruit Art | 'Fruit Art', 'Decorative Fruit Dessert', 'Mango and Blueberry Desserts' | 0.851 | SEMANTIC |
| 816 | Noodles with Egg | Noodles with Boiled Egg | 'Noodles with Boiled Egg', 'Salted Egg Yolk Noodles', 'Noodle dish with boiled eggs' | 0.958 | SEMANTIC |
| 817 | Stir-fried Chicken with Vegetables | Chili Crab | 'Chili Crab', 'Mixed Chinese Seafood and Meat Dishes', 'Stir-fried seafood and meat with bread' | 0.669 | MISS |
| 818 | Sushi Platter | Sashimi or Raw Seafood Platter | 'Sashimi or Raw Seafood Platter', 'Sushi and Grilled Fish Platter', 'Sashimi Platter with Rice' | 0.851 | SEMANTIC |
| 819 | Apple | Peeled Apple | 'Peeled Apple', 'Asian Pear', 'Green Apple' | 0.867 | SEMANTIC |
| 820 | Egg | Egg | 'Egg', 'Tamago', 'Spiced Egg' | 1.000 | EXACT |
| 821 | Salad and Pizza | Pizza and Salad | 'Pizza and Salad', 'Salad and Pizza', 'Pizza with Salad' | 0.970 | SEMANTIC |
| 822 | Stir-fried Chicken with Vegetables | Kung Pao Chicken | 'Kung Pao Chicken', 'Sweet and Sour Chicken with Corn', 'Sichuan Chicken' | 0.785 | SEMANTIC |
| 823 | Steamed Fish with Rice and Vegetables | Steamed Fish with Rice and Salad | 'Steamed Fish with Rice and Salad', 'Steamed Fish in Soy Sauce', 'Steamed Fish with Side Dishes' | 0.914 | SEMANTIC |
| 824 | Orange | Mandarin Orange | 'Mandarin Orange', 'Orange Slice', 'Orange slice' | 0.837 | SEMANTIC |
| 825 | Grilled Squid | Fried Skewered Food | 'Fried Skewered Food', 'Fried Insects', 'Skewered Street Food' | 0.715 | SEMANTIC |
| 826 | Braised Pork | Braised Chicken/ Duck | 'Braised Chicken/ Duck', 'Braised Duck with Rice', 'Spicy Braised Duck' | 0.914 | SEMANTIC |
| 827 | Braised Pork | Octopus-shaped sausages | 'Octopus-shaped sausages', 'Sausage Squid', 'Chicken Feet and Meat' | 0.698 | MISS |
| 828 | Cheeseburger | Cheeseburger Bagel | 'Cheeseburger Bagel', 'Beef Bagel Sandwich', 'Bagel Burger' | 0.844 | SEMANTIC |