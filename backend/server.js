// server.js
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const winston = require('winston');
require('dotenv').config();

// Import routes
const userRoutes = require('./routes/userRoutes');
const foodRoutes = require('./routes/foodRoutes');
const accountRoutes = require("./routes/accountRoutes");
const exerciseRoutes = require("./routes/exerciseRoutes");
const chatRoutes = require('./routes/chatRoutes')
const app = express();

// ---------------------
// 🔹 Logger cấu hình bằng Winston
// ---------------------
const logger = winston.createLogger({

    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
    ),
    transports: [
        new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
        new winston.transports.Console({ format: winston.format.simple() })
    ]
});


app.use(cors({
  origin: ['http://localhost:5000', 'http://192.168.30.244:5000', 'http://42.112.64.53:5000'],
  credentials: true
}));

app.use(express.json());
app.use((req, res, next) => {
    logger.info(`${req.method} ${req.url} - Body: ${JSON.stringify(req.body)} - Query: ${JSON.stringify(req.query)}`);
    next();
});

// ---------------------
// 🔹 Routes
// ---------------------
app.use('/api/users', userRoutes);
app.use('/api/foods', foodRoutes);
app.use("/api/accounts", accountRoutes);
app.use("/api/exercises", exerciseRoutes);
app.use('/api/chat',chatRoutes)
// Endpoint kiểm tra server
app.get('/health', (req, res) => {
    res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// ---------------------
// 🔹 Middleware xử lý lỗi
// ---------------------
app.use((err, req, res, next) => {
    logger.error(err.stack);
    res.status(500).json({ error: 'Internal server error', details: err.message });
});

// ---------------------
// 🔹 Kết nối MongoDB
// ---------------------
mongoose.connect(process.env.MONGO_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true
})
.then(() => console.log('✅ MongoDB connected successfully'))
.catch(err => console.error('❌ MongoDB connection error:', err));

// ---------------------
// 🔹 Khởi động Server
// ---------------------
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));

