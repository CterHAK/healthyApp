# 📊 Class Diagram - Healthy App Project

## 🎯 Project Overview
Ứng dụng sức khỏe toàn diện gồm 3 phần: **Frontend (React Native)**, **Backend (Node.js)**, **AI Service (Python)**

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND (React Native - MyApp)               │
├─────────────────────────────────────────────────────────────────┤
│  • User Interface (Screens & Components)                         │
│  • Local User Model                                              │
│  • API Services (HTTP Client)                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP REST API
┌────────────────────────────▼────────────────────────────────────┐
│                   BACKEND (Node.js - Express)                   │
├─────────────────────────────────────────────────────────────────┤
│  • Controllers (Request Handlers)                                │
│  • Services (Business Logic)                                     │
│  • MongoDB Models & Schemas                                      │
│  • Routes (API Endpoints)                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │ Python Subprocess
┌────────────────────────────▼────────────────────────────────────┐
│             AI SERVICE (Python - healthyApp)                     │
├─────────────────────────────────────────────────────────────────┤
│  • Food Recognition (CLIP Model)                                 │
│  • Exercise Planning (Filtering)                                 │
│  • Nutrition Analysis                                            │
│  • Meal Planning (RAG-based)                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ BACKEND - Class Diagram

### 1️⃣ **DATABASE MODELS**

#### 📌 User Model
```
┌──────────────────────────────────────┐
│         User (MongoDB Schema)        │
├──────────────────────────────────────┤
│ - email: String (unique)             │
│ - name: String                       │
│ - gender: String                     │
│ - age: Number                        │
│ - height: Number (cm)                │
│ - weight: Number (kg)                │
│ - targetWeight: Number               │
│ - target: String                     │
│ - exercise: String                   │
│ - allergies: [String]                │
│ - diseases: [String]                 │
│ - caloriePlan: String                │
│ - bmr: Number                        │
│ - tdee: Number                       │
├──────────────────────────────────────┤
│ Methods:                             │
│ + create()                           │
│ + findByEmail(email)                 │
│ + updateUser(email, data)            │
│ + delete(email)                      │
└──────────────────────────────────────┘
```

#### 📌 Account Model
```
┌──────────────────────────────────────┐
│       Account (MongoDB Schema)       │
├──────────────────────────────────────┤
│ - email: String (unique, required)   │
│ - password: String (hashed)          │
├──────────────────────────────────────┤
│ Methods:                             │
│ + register(email, password)          │
│ + login(email, password)             │
│ + validatePassword(password)         │
│ + findByEmail(email)                 │
│ + updatePassword(email, newPass)     │
│ + delete(email)                      │
└──────────────────────────────────────┘
```

#### 📌 Food Model
```
┌──────────────────────────────────────┐
│        Food (MongoDB Schema)         │
├──────────────────────────────────────┤
│ - email: String (required)           │
│ - dish_name: String (required)       │
│ - ingredients: [String]              │
│ - portion_size: String               │
│ - nutrition:                         │
│   • calories_kcal: Number            │
│   • protein_g: Number                │
│   • carbohydrate_g: Number           │
│   • fat_g: Number                    │
│ - image_url: String                  │
│ - day: String (DD-MM-YYYY)          │
│ - session: String (Sáng/Trưa/Tối)   │
│ - is_recognized: Boolean             │
│ - createdAt: Date                    │
│ - updatedAt: Date                    │
├──────────────────────────────────────┤
│ Methods:                             │
│ + create(foodData)                   │
│ + findByEmailAndDay(email, day)      │
│ + findAll(email)                     │
│ + update(id, data)                   │
│ + delete(id)                         │
│ + getTotalNutrition(email, day)      │
└──────────────────────────────────────┘
```

#### 📌 ExerciseGroup Model
```
┌──────────────────────────────────────┐
│    ExerciseGroup (MongoDB Schema)    │
├──────────────────────────────────────┤
│ - email: String (required)           │
│ - group_name: String (required)      │
├──────────────────────────────────────┤
│ Methods:                             │
│ + create(email, groupName)           │
│ + findByEmail(email)                 │
│ + findByGroupName(groupName)         │
│ + update(groupName, data)            │
│ + delete(groupName)                  │
└──────────────────────────────────────┘
   │ 1..∞
   │ has
   ▼
┌──────────────────────────────────────┐
│   ExerciseDetail (MongoDB Schema)    │
├──────────────────────────────────────┤
│ - email: String (required)           │
│ - group_name: String (required)      │
│ - name: String (required)            │
│ - reps: Number (default: 10)         │
│ - sets: Number (default: 3)          │
├──────────────────────────────────────┤
│ Methods:                             │
│ + create(data)                       │
│ + findByGroupName(groupName)         │
│ + update(id, data)                   │
│ + delete(id)                         │
│ + getTotalVolume()                   │
└──────────────────────────────────────┘
```

#### 📌 WorkoutPlan Model
```
┌──────────────────────────────────────┐
│     WorkoutPlan (MongoDB Schema)     │
├──────────────────────────────────────┤
│ - email: String (required)           │
│ - group_name: String (required)      │
│ - day: String (required)             │
│ - session: String (required)         │
│ - done_flag: Boolean (default: false)│
├──────────────────────────────────────┤
│ Methods:                             │
│ + create(data)                       │
│ + findByEmailAndDay(email, day)      │
│ + markAsComplete(id)                 │
│ + update(id, data)                   │
│ + delete(id)                         │
│ + getProgress(email)                 │
└──────────────────────────────────────┘
   │ references
   └──▶ ExerciseDetail
   │ references
   └──▶ ExerciseGroup
```

---

### 2️⃣ **SERVICE LAYER**

#### 📌 UserService
```
┌────────────────────────────────────────┐
│         UserService                    │
├────────────────────────────────────────┤
│ Methods:                               │
│ + createUser(userData): Promise<User>  │
│ + getAllUsers(): Promise<[User]>       │
│ + getUserByEmail(email): Promise<User> │
│ + updateUser(email, data): Promise<?>  │
│ + deleteUser(email): Promise<>         │
│ + calculateBMR(weight, height, age)    │
│ + calculateTDEE(bmr, activity)         │
└────────────────────────────────────────┘
   │ uses
   ├──▶ User Model
   └──▶ Account Model
```

#### 📌 FoodService
```
┌────────────────────────────────────────┐
│         FoodService                    │
├────────────────────────────────────────┤
│ Methods:                               │
│ + saveFood(foodData): Promise<Food>    │
│ + getFoodByEmail(email): Promise<[F]>  │
│ + getFoodByDay(email, day): Pr<[F]>   │
│ + updateFood(id, data): Promise<Food>  │
│ + deleteFood(id): Promise<>            │
│ + getTotalNutrition(email, day)        │
│ + generateNutritionReport(email, d)    │
│ + getRecognizedVsManual(email)         │
└────────────────────────────────────────┘
   │ uses
   └──▶ Food Model
```

#### 📌 AccountService
```
┌────────────────────────────────────────┐
│       AccountService                   │
├────────────────────────────────────────┤
│ Methods:                               │
│ + register(email, pass): Promise<Acc>  │
│ + login(email, pass): Promise<Acc>     │
│ + verifyPassword(email, pass): bool    │
│ + updatePassword(email, newPass)       │
│ + deleteAccount(email): Promise<>      │
│ + hashPassword(pass): String           │
│ + generateToken(email): String         │
└────────────────────────────────────────┘
   │ uses
   └──▶ Account Model
```

#### 📌 ExerciseService
```
┌────────────────────────────────────────┐
│      ExerciseService                   │
├────────────────────────────────────────┤
│ Methods:                               │
│ + createGroup(email, name): Promise    │
│ + addExercise(groupName, data)         │
│ + getGroupByEmail(email): Promise<[G]>│
│ + getExercisesByGroup(gName): Pr<[E]> │
│ + updateExercise(id, data)             │
│ + deleteExercise(id)                   │
│ + calculateTotalVolume(groupName)      │
└────────────────────────────────────────┘
   │ uses
   ├──▶ ExerciseGroup Model
   └──▶ ExerciseDetail Model
```

#### 📌 PythonService
```
┌────────────────────────────────────────┐
│       PythonService                    │
├────────────────────────────────────────┤
│ - PYTHON_SCRIPT: String                │
│ Methods:                               │
│ + runPython(inputData): Promise<JSON>  │
│ + recognizeFood(imagePath): Pr<Food>   │
│ + planExercise(userData): Pr<Plan>     │
│ + analyzeMealPlan(meals): Pr<Analysis> │
│ + generateRecommendation(user): Pr<R> │
└────────────────────────────────────────┘
   │ spawns subprocess
   └──▶ Python Process
        ├──▶ food_api_script.py
        ├──▶ exercise_api_script.py
        ├──▶ food_recognition_service.py
        └──▶ meal_planning_service.py
```

---

### 3️⃣ **CONTROLLER LAYER**

#### 📌 UserController
```
┌────────────────────────────────────────┐
│      UserController                    │
├────────────────────────────────────────┤
│ Methods:                               │
│ + createUser(req, res): void           │
│ + getUsers(req, res): void             │
│ + getUserByEmail(req, res): void       │
│ + updateUser(req, res): void           │
│ + deleteUser(req, res): void           │
└────────────────────────────────────────┘
   │ uses
   └──▶ UserService
```

#### 📌 FoodController
```
┌────────────────────────────────────────┐
│      FoodController                    │
├────────────────────────────────────────┤
│ Methods:                               │
│ + searchFood(req, res): void (SSE)     │
│ + recognizeFood(req, res): void        │
│ + saveFood(req, res): void             │
│ + getFoods(req, res): void             │
│ + updateFood(req, res): void           │
│ + deleteFood(req, res): void           │
│ + getFoodByDay(req, res): void         │
│ + getNutritionReport(req, res): void   │
└────────────────────────────────────────┘
   │ uses
   ├──▶ FoodService
   └──▶ PythonService
```

#### 📌 ExerciseController
```
┌────────────────────────────────────────┐
│    ExerciseController                  │
├────────────────────────────────────────┤
│ Methods:                               │
│ + createGroup(req, res): void          │
│ + addExercise(req, res): void          │
│ + getGroups(req, res): void            │
│ + getExercises(req, res): void         │
│ + updateExercise(req, res): void       │
│ + deleteExercise(req, res): void       │
│ + planWorkout(req, res): void          │
│ + completeWorkout(req, res): void      │
└────────────────────────────────────────┘
   │ uses
   ├──▶ ExerciseService
   └──▶ PythonService
```

#### 📌 ChatController
```
┌────────────────────────────────────────┐
│      ChatController                    │
├────────────────────────────────────────┤
│ Methods:                               │
│ + sendMessage(req, res): void (SSE)    │
│ + getHistory(req, res): void           │
│ + clearHistory(req, res): void         │
└────────────────────────────────────────┘
   │ uses
   └──▶ ChatService
```

---

### 4️⃣ **ROUTES**

```
┌──────────────────────────────────────────────────┐
│              API Routes                          │
├──────────────────────────────────────────────────┤
│ /api/users/         → UserController             │
│   GET    /          → getUsers()                 │
│   GET    /:email    → getUserByEmail()           │
│   POST   /          → createUser()               │
│   PUT    /:email    → updateUser()               │
│   DELETE /:email    → deleteUser()               │
├──────────────────────────────────────────────────┤
│ /api/foods/         → FoodController             │
│   GET    /          → getFoods()                 │
│   GET    /day/:day  → getFoodByDay()             │
│   POST   /          → saveFood()                 │
│   POST   /recognize → recognizeFood()            │
│   PUT    /:id       → updateFood()               │
│   DELETE /:id       → deleteFood()               │
├──────────────────────────────────────────────────┤
│ /api/exercises/     → ExerciseController         │
│   GET    /groups    → getGroups()                │
│   POST   /groups    → createGroup()              │
│   GET    /details   → getExercises()             │
│   POST   /details   → addExercise()              │
│   PUT    /:id       → updateExercise()           │
│   DELETE /:id       → deleteExercise()           │
├──────────────────────────────────────────────────┤
│ /api/chat/          → ChatController             │
│   POST   /message   → sendMessage() (SSE)        │
│   GET    /history   → getHistory()               │
├──────────────────────────────────────────────────┤
│ /api/nutrition/     → NutritionController        │
│   GET    /report    → getNutritionReport()       │
│   GET    /analysis  → analyzeNutrition()         │
└──────────────────────────────────────────────────┘
```

---

## 🎨 FRONTEND - Class Diagram

### 1️⃣ **User Model (Local)**

```
┌────────────────────────────────────────┐
│         User (JavaScript Class)        │
├────────────────────────────────────────┤
│ Properties:                            │
│ - email: String                        │
│ - password: String                     │
│ - name: String                         │
│ - gender: String                       │
│ - age: Number                          │
│ - height: Number                       │
│ - weight: Number                       │
│ - targetWeight: Number                 │
│ - target: String                       │
│ - exercise: String                     │
│ - allergies: [String]                  │
│ - diseases: [String]                   │
│ - caloriePlan: String                  │
│ - bmr: Number                          │
│ - tdee: Number                         │
├────────────────────────────────────────┤
│ Methods:                               │
│ + constructor(data)                    │
│ + updateField(key, value)              │
│ + toJSON(): Object                     │
│ + isValid(): Boolean                   │
│ + calculateBMI(): Number               │
└────────────────────────────────────────┘
```

### 2️⃣ **API Services**

#### 📌 AccountAPI
```
┌────────────────────────────────────────┐
│        AccountAPI (HTTP Service)       │
├────────────────────────────────────────┤
│ - BASE_URL: String                     │
│ Methods:                               │
│ + register(email, pass): Promise       │
│ + login(email, pass): Promise<Token>   │
│ + logout(): Promise                    │
│ + updateProfile(data): Promise         │
│ + changePassword(newPass): Promise     │
└────────────────────────────────────────┘
```

#### 📌 UserAPI
```
┌────────────────────────────────────────┐
│         UserAPI (HTTP Service)         │
├────────────────────────────────────────┤
│ Methods:                               │
│ + getUser(email): Promise<User>        │
│ + updateUser(email, data): Promise     │
│ + getUserStats(email): Promise<Stats>  │
│ + calculateNutrition(foods): Promise   │
└────────────────────────────────────────┘
```

#### 📌 FoodAPI
```
┌────────────────────────────────────────┐
│         FoodAPI (HTTP Service)         │
├────────────────────────────────────────┤
│ Methods:                               │
│ + getFoods(email): Promise<[Food]>     │
│ + getFoodByDay(email, day): Pr<[F]>   │
│ + saveFood(data): Promise<Food>        │
│ + updateFood(id, data): Promise        │
│ + deleteFood(id): Promise              │
│ + getTotalNutrition(email, day)        │
│ + getFoodReport(email): Promise        │
└────────────────────────────────────────┘
```

#### 📌 FoodRecognitionAPI
```
┌────────────────────────────────────────┐
│    FoodRecognitionAPI (HTTP Service)   │
├────────────────────────────────────────┤
│ Methods:                               │
│ + recognizeFood(imagePath):Promise<F>  │
│ + saveFoodFromRecognition(data):Pr<F> │
│ + batchRecognize([images]): Promise    │
└────────────────────────────────────────┘
```

#### 📌 ExerciseAPI
```
┌────────────────────────────────────────┐
│       ExerciseAPI (HTTP Service)       │
├────────────────────────────────────────┤
│ Methods:                               │
│ + getGroups(email): Promise<[Group]>   │
│ + createGroup(name): Promise<Group>    │
│ + addExercise(data): Promise<Exercise> │
│ + getExercises(groupName): Promise     │
│ + updateExercise(id, data): Promise    │
│ + deleteExercise(id): Promise          │
│ + planWorkout(email): Promise<Plan>    │
└────────────────────────────────────────┘
```

#### 📌 ChatAPI
```
┌────────────────────────────────────────┐
│        ChatAPI (HTTP Service - SSE)    │
├────────────────────────────────────────┤
│ Methods:                               │
│ + sendMessage(msg): EventSource        │
│ + getHistory(): Promise<[Messages]>    │
│ + clearHistory(): Promise              │
│ + parseSSEResponse(event): Object      │
└────────────────────────────────────────┘
```

#### 📌 CloudinaryAPI
```
┌────────────────────────────────────────┐
│      CloudinaryAPI (File Upload)       │
├────────────────────────────────────────┤
│ Methods:                               │
│ + uploadImage(file): Promise<URL>      │
│ + uploadImages([files]): Promise<[URL]>│
│ + deleteImage(url): Promise            │
│ + resizeImage(url, size): String       │
└────────────────────────────────────────┘
```

### 3️⃣ **Screen Components**

```
┌────────────────────────────────────────┐
│      LoginScreen                       │
├────────────────────────────────────────┤
│ Props:                                 │
│ - navigation: NavigationProp           │
│ State:                                 │
│ - email, password: String              │
│ - isLoading: Boolean                   │
│ Methods:                               │
│ + handleLogin(): Promise               │
│ + handleRegister(): Promise            │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│      RegisterScreen                    │
├────────────────────────────────────────┤
│ Props:                                 │
│ - navigation: NavigationProp           │
│ State:                                 │
│ - formData: User                       │
│ - errors: [String]                     │
│ Methods:                               │
│ + handleRegister(): Promise            │
│ + validateForm(): Boolean              │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│    FoodCaptureScreen                   │
├────────────────────────────────────────┤
│ Props:                                 │
│ - route, navigation                    │
│ State:                                 │
│ - image: URI                           │
│ - dishName: String                     │
│ - ingredients: [String]                │
│ - nutrition: {cal, prot, carb, fat}    │
│ - mealType: String                     │
│ - isRecognizing: Boolean               │
│ Methods:                               │
│ + pickImage(): Promise                 │
│ + autoRecognizeFood(): Promise         │
│ + addIngredient(): void                │
│ + removeIngredient(idx): void          │
│ + saveToBackend(): Promise             │
│ + updateNutrition(field, val): void    │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│    ExerciseGroupAddScreen              │
├────────────────────────────────────────┤
│ Props:                                 │
│ - route, navigation                    │
│ State:                                 │
│ - groupName: String                    │
│ - exercises: [{name, reps, sets}]      │
│ - isLoading: Boolean                   │
│ Methods:                               │
│ + addExerciseGroup(): Promise          │
│ + addExercise(): void                  │
│ + removeExercise(idx): void            │
│ + saveWorkout(): Promise               │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│    DashBoardScreen                     │
├────────────────────────────────────────┤
│ Props:                                 │
│ - route, navigation                    │
│ State:                                 │
│ - userData: User                       │
│ - todayNutrition: Object               │
│ - todayExercise: Object                │
│ - widgets: [Widget]                    │
│ Methods:                               │
│ + loadUserData(): Promise              │
│ + calculateStats(): Object             │
│ + refreshDashboard(): Promise          │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│    ChatBotScreen                       │
├────────────────────────────────────────┤
│ Props:                                 │
│ - route, navigation                    │
│ State:                                 │
│ - messages: [{user, bot}]              │
│ - inputText: String                    │
│ - isLoading: Boolean                   │
│ Methods:                               │
│ + sendMessage(): Promise               │
│ + onSSEMessage(event): void            │
│ + loadHistory(): Promise               │
│ + clearChat(): Promise                 │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│    FoodHistoryScreen                   │
├────────────────────────────────────────┤
│ Props:                                 │
│ - route, navigation                    │
│ State:                                 │
│ - foods: [Food]                        │
│ - selectedDate: Date                   │
│ - filterType: String                   │
│ Methods:                               │
│ + loadFoodHistory(): Promise           │
│ + filterFoods(type): [Food]            │
│ + deleteFood(id): Promise              │
│ + editFood(id): void                   │
└────────────────────────────────────────┘
```

---

## 🐍 AI SERVICE - Python Class Diagram

### 1️⃣ **Food Recognition Module**

```
┌────────────────────────────────────────┐
│      FoodRecognizer (Python Class)     │
├────────────────────────────────────────┤
│ Attributes:                            │
│ - device: str (cuda/cpu)               │
│ - image_path: str                      │
│ - raw_image: Image                     │
│ - clip_model: CLIPModel                │
│ - clip_processor: CLIPProcessor        │
│ - df: DataFrame                        │
│ - candidate_labels: [str]              │
│ - label_embeddings: Tensor             │
│ - batch_size: int                      │
├────────────────────────────────────────┤
│ Methods:                               │
│ + _load_image(path): PIL.Image         │
│ + prepare_dataset(): void              │
│ + classify_image(): dict               │
│ + get_image_embedding(): Tensor        │
│ + compute_text_embeddings(): Tensor    │
│ + _extract_nutrition_info(label): dict │
│ + recognize(): dict                    │
└────────────────────────────────────────┘
   │ uses
   ├──▶ CLIP Model (openai/clip-vit)
   ├──▶ MM-Food-100K Dataset
   └──▶ FAISS Index
```

### 2️⃣ **Food Database Module**

```
┌────────────────────────────────────────┐
│      FoodDatabase (Python Class)       │
├────────────────────────────────────────┤
│ Attributes:                            │
│ - food_data: DataFrame                 │
│ - faiss_index: FAISS                   │
│ - embeddings: np.Array                 │
│ - food_list: [str]                     │
├────────────────────────────────────────┤
│ Methods:                               │
│ + load_food_data(csv): void            │
│ + build_faiss_index(): void            │
│ + search(query, k): [(str, float)]     │
│ + get_nutrition(food_name): dict       │
│ + get_similar_foods(food, k): [str]    │
│ + get_alternatives(food): [str]        │
└────────────────────────────────────────┘
   │ uses
   ├──▶ FAISS Vectors
   ├──▶ Food CSV Data
   └──▶ Embeddings
```

### 3️⃣ **Meal Planning Module**

```
┌────────────────────────────────────────┐
│    MealPlanner (Python Class)          │
├────────────────────────────────────────┤
│ Attributes:                            │
│ - food_database: FoodDatabase          │
│ - user_profile: dict                   │
│ - daily_calories: float                │
│ - macro_targets: dict                  │
│ - rag_retriever: RAGRetriever          │
├────────────────────────────────────────┤
│ Methods:                               │
│ + set_daily_calories(cal): void        │
│ + set_macro_targets(p, c, f): void     │
│ + generate_meal_plan(days): Plan       │
│ + optimize_meals(): Plan               │
│ + get_recommendations(): [Food]        │
│ + calculate_macro_balance(): dict      │
│ + suggest_alternatives(food): [Food]   │
└────────────────────────────────────────┘
   │ uses
   ├──▶ FoodDatabase
   └──▶ RAG Retriever
```

### 4️⃣ **Exercise Planning Module**

```
┌────────────────────────────────────────┐
│     ExerciseFilter (Python Class)      │
├────────────────────────────────────────┤
│ Attributes:                            │
│ - exercises_data: DataFrame            │
│ - user_profile: dict                   │
│ - fitness_level: str                   │
│ - goals: [str]                         │
│ - available_time: float (minutes)      │
├────────────────────────────────────────┤
│ Methods:                               │
│ + load_exercises(csv): void            │
│ + filter_by_muscle(muscle): [Exercise] │
│ + filter_by_intensity(level): [Ex]    │
│ + filter_by_time(minutes): [Exercise]  │
│ + generate_workout_plan(): Plan        │
│ + get_alternatives(exercise): [Ex]     │
│ + adjust_difficulty(plan, level)       │
│ + calculate_total_time(): int          │
└────────────────────────────────────────┘
   │ uses
   └──▶ Exercises CSV Data
```

### 5️⃣ **Nutrition Analysis Module**

```
┌────────────────────────────────────────┐
│    NutritionAnalyzer (Python Class)    │
├────────────────────────────────────────┤
│ Attributes:                            │
│ - user_data: dict                      │
│ - meals_history: [Food]                │
│ - daily_goals: dict                    │
│ - health_conditions: [str]             │
├────────────────────────────────────────┤
│ Methods:                               │
│ + calculate_bmr(w, h, a, g): float    │
│ + calculate_tdee(bmr, activity): flt   │
│ + analyze_nutrition_intake(): dict     │
│ + get_macro_breakdown(): dict          │
│ + check_allergies(foods): [str]        │
│ + get_nutritional_advice(): str        │
│ + generate_report(period): Report      │
│ + get_deficit_foods(): [Food]          │
└────────────────────────────────────────┘
   │ uses
   ├──▶ FoodDatabase
   └──▶ Nutrition Converter
```

### 6️⃣ **Utility Modules**

```
┌────────────────────────────────────────┐
│      FAISSUtils (Helper)               │
├────────────────────────────────────────┤
│ Functions:                             │
│ + build_index([embeddings]): FAISS     │
│ + search_index(index, query, k): res   │
│ + save_index(index, path): void        │
│ + load_index(path): FAISS              │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│    NutritionUtils (Helper)             │
├────────────────────────────────────────┤
│ Functions:                             │
│ + convert_calories(g, type): float     │
│ + calculate_macro_ratio(p,c,f): dict   │
│ + validate_nutrition_data(data): bool  │
│ + calculate_percentage(val, total)     │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│     DataPreparation (Helper)           │
├────────────────────────────────────────┤
│ Functions:                             │
│ + clean_food_data(df): DataFrame       │
│ + add_embeddings(df): DataFrame        │
│ + load_data(path): DataFrame           │
│ + prepare_rag_data(df): RAGData        │
└────────────────────────────────────────┘
```

---

## 🔄 **Integration Relationships**

### **Request Flow Example: Food Recognition**

```
1. FoodCaptureScreen (Frontend)
   ↓ pickImage()
2. ImagePicker (Expo)
   ↓ image selected
3. uploadImageToCloudinary()
   ↓ image URL
4. recognizeFood(imageUrl) [FoodRecognitionAPI]
   ↓ POST /foods/recognize
5. FoodController.recognizeFood()
   ↓ uses
6. PythonService.runPython()
   ↓ spawns
7. food_recognition_service.py
   ↓ uses
8. FoodRecognizer.recognize()
   ↓ CLIP model inference
9. Response with {dish_name, ingredients, nutrition}
   ↓
10. FoodCaptureScreen displays recognized data
    ↓ user confirms
11. saveToBackend() [FoodAPI.saveFood()]
    ↓ POST /foods
12. FoodController.saveFood()
    ↓ uses
13. FoodService.saveFood()
    ↓ save
14. Food Document stored in MongoDB
```

### **Meal Planning Flow**

```
User Input (User Profile + Daily Goal)
   ↓
NutritionService.calculateTDEE()
   ↓
MealPlanner.generate_meal_plan()
   ├──▶ FoodDatabase.search()
   ├──▶ RAG Retriever search for preferences
   └──▶ Optimization algorithm
   ↓
Plan with recommended meals
   ↓
Backend returns to Frontend
   ↓
Display meal suggestions on DashBoard
```

---

## 📊 **Data Flow Architecture**

```
┌─────────────────────────────────────────────────────────┐
│                 React Native App (Frontend)              │
│                    (Screens & Components)                │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP REST + SSE
┌────────────────────────▼────────────────────────────────┐
│              Express.js Backend                          │
│        (Controllers → Services → Models)                 │
├─────────────────────────────────────────────────────────┤
│              Routes:                                     │
│  /api/users       /api/foods      /api/exercises        │
│  /api/chat        /api/nutrition  /api/accounts         │
└────────────────┬──────────────────────────┬─────────────┘
                 │                          │
    ┌────────────▼──────────────┐    ┌──────▼─────────────┐
    │    MongoDB Database       │    │  Python Services   │
    │  (Users, Foods, Exer...)  │    │ (AI, ML, Analysis) │
    └──────────────────────────┘    └────────────────────┘
         Collections:                   Modules:
         • Users                        • FoodRecognizer
         • Accounts                     • MealPlanner
         • Foods                        • ExerciseFilter
         • ExerciseGroups               • NutritionAnalyzer
         • ExerciseDetails              • DataPreparation
         • WorkoutPlans
```

---

## 🎯 **Key Design Patterns**

### 1. **Service Layer Pattern**
- Business logic separated from controllers
- Reusable across multiple endpoints

### 2. **Model-View-Controller (MVC)**
- Backend: Controllers handle requests → Services process → Models store
- Frontend: Screens as Views → API Services as Controllers → Local User model

### 3. **Microservice Pattern (Python Integration)**
- Python runs as subprocess via Node.js
- Independent AI/ML logic
- Scalable for heavy computations

### 4. **Factory Pattern**
- Model creation through service methods
- Centralized object instantiation

### 5. **Repository Pattern**
- MongoDB models act as repositories
- Abstract data access from business logic

### 6. **Observer Pattern (Frontend)**
- State management updates screen re-renders
- SSE for real-time chat updates

---

## 📈 **Scalability Considerations**

```
Current:
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Frontend   │      │   Backend    │      │  Python AI   │
│ (React N.)   │◄────►│  (Node.js)   │◄────►│  (subprocess)│
└──────────────┘      └──────────────┘      └──────────────┘
                             │
                      ┌──────▼────────┐
                      │   MongoDB     │
                      │   Database    │
                      └───────────────┘

Suggested Future Improvements:
1. Message Queue (Redis) for async operations
2. Separate Python API Service (FastAPI/Flask)
3. Caching Layer (Redis) for frequent queries
4. Load Balancing for multiple backend instances
5. WebSocket instead of SSE for chat
6. API Gateway for routing
```

---

## 🔐 **Security Architecture**

```
┌─────────────────────────────────────────────┐
│         Frontend Security                   │
├─────────────────────────────────────────────┤
│ • Token storage in secure storage           │
│ • JWT token validation                      │
│ • HTTPS only communication                  │
│ • Input validation                          │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│         Backend Security                    │
├─────────────────────────────────────────────┤
│ • JWT middleware authentication             │
│ • Password hashing (bcrypt)                 │
│ • Input sanitization                        │
│ • Rate limiting                             │
│ • CORS policy                               │
│ • Error handling (no stack traces)          │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│         Database Security                   │
├─────────────────────────────────────────────┤
│ • MongoDB user authentication               │
│ • Collection-level access control           │
│ • Encrypted sensitive fields                │
│ • Indexed queries for performance           │
└─────────────────────────────────────────────┘
```

---

## 📋 **Summary Table**

| Layer | Component | Technology | Responsibility |
|-------|-----------|-----------|-----------------|
| **Frontend** | Screens | React Native | User Interface |
| **Frontend** | Services | JavaScript HTTP | API Communication |
| **Frontend** | Models | JavaScript Classes | Local Data |
| **Backend** | Controllers | Express.js | Request Handling |
| **Backend** | Services | Node.js | Business Logic |
| **Backend** | Models | Mongoose | Data Persistence |
| **Database** | Collections | MongoDB | Data Storage |
| **AI/ML** | Recognizer | Python + CLIP | Food Recognition |
| **AI/ML** | Planner | Python + RAG | Meal Planning |
| **AI/ML** | Analyzer | Python + Utils | Analysis |

---

**Created:** December 3, 2025
**Project:** Healthy App - Comprehensive Health & Fitness Management System

