const express = require('express')
// const {GoogleGenerativeAI} = require('@google/generative-ai');
// require('dotenv').config();

const app = express();

// const genAI = new GoogleGenerativeAI(process.env.API_KEY);

// const model = genAI.getGenerativeModel({model:'gemini-pro'})

const PORT = 8080;

// Router functionalities

const summary_route = require('./routers/resume-summary');


app.use('/summary',summary_route);

app.get('/',(req,res)=>{

    res.send("Hello World!");
})

app.listen(PORT,()=>{
    console.log(`Server running on http://localhost:8080`);
})