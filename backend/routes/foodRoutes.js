const express = require('express');
const router = express.Router();
const foodController = require('../controllers/foodController');
const rateLimit = require('express-rate-limit');

const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100,
    message: { error: 'Too many requests, please try again later.' }
});

const validateSearch = (req, res, next) => {
    const { q } = req.query;
    if (!q || typeof q !== 'string' || q.trim().length === 0) {
        return res.status(400).json({ error: "Query parameter 'q' must be a non-empty string" });
    }
    next();
};

router.get('/search', validateSearch, limiter, foodController.searchFood);

module.exports = router;