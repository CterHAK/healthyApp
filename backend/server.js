// server.js
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const winston = require('winston');
require('dotenv').config();

const userRoutes = require('./routes/userRoutes');
const foodRoutes = require('./routes/foodRoutes');

const app = express();

// Logger
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
    ),
    transports: [new winston.transports.File({ filename: 'logs/error.log', level: 'error' })]
});

// Middleware
app.use(cors({ origin: 'http://localhost:3000', credentials: true }));
app.use(express.json());
app.use((req, res, next) => {
    logger.info(`${req.method} ${req.url} - Body: ${JSON.stringify(req.body)} - Query: ${JSON.stringify(req.query)}`);
    next();
});

// Routes
app.use('/users', userRoutes);
app.use('/api', foodRoutes);
app.get('/health', (req, res) => {
    res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// Error Middleware
app.use((err, req, res, next) => {
    logger.error(err.stack);
    res.status(500).json({ error: 'Internal server error', details: err.message });
});

// MongoDB Connection
mongoose.connect(process.env.MONGO_URI)
    .then(() => console.log('✅ MongoDB connected successfully'))
    .catch(err => console.error('❌ MongoDB connection error:', err));

// Start Server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));