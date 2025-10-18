const express = require('express');
const bodyParser = require('body-parser');
const userRoutes = require('./routes/userRoutes');
const foodRoutes = require('./routes/foodRoutes');

const app = express();

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

app.use('/api', userRoutes);
app.use('/api', foodRoutes);

app.listen(5000, () => {
  console.log('Server is running on port 5000');
});