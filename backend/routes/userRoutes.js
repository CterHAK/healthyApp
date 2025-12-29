const express = require("express");
const router = express.Router();
const userController = require("../controllers/userController");

// POST /api/users
router.post("/", userController.createUser);

// GET /api/users
router.get("/", userController.getUsers);

// GET /api/users/:email
router.get("/:email", userController.getUserByEmail);

module.exports = router;
