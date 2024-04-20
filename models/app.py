from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flask_cors import CORS
import pdfplumber
from helper import *
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv('API_KEY'))

app = Flask(__name__)
CORS(app)

# load model

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=25) 
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased") 
model.load_state_dict(torch.load('pretrained_model.pth', map_location=torch.device('cpu')))
model.eval()

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/similarity',methods=['POST'])
def similarity():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file part'})
    
    
    resume = request.files['resume']

    text = parse_pdf(resume)
    text = preprocess_text(text)
    data = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
    prediction = model(**data).logits.detach().numpy().tolist()[0]
    
    labels = np.argsort(prediction)[::-1]
    
    predictions = [reverse_label_encoder[label] for label in labels[:5]]
    return jsonify({'result':predictions})

@app.route('/optimise',methods=['POST'])
def optimise():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file part'})
    
    
    resume = request.files['resume']

    text = parse_pdf(resume)
    text = preprocess_text(text)

    text = 'What should I do to optimise this resume? \n' + text
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(text)
    return jsonify({"text" : response.text})
    

@app.route('/predict', methods=['POST'])
def predict():
    # get data from request
    '''
        Request format:
        {
            "data": "This is a sample text"
        }
    '''

    if 'resume' not in request.files:
        return jsonify({'error': 'No file part'})
    
    
    resume = request.files['resume']

    text = parse_pdf(resume)
    text = preprocess_text(text)

    print(text)
    # make prediction
    data = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
    prediction = model(**data).logits.detach().numpy().tolist()[0]
    
    labels = np.argsort(prediction)[::-1]
    
    predictions = [reverse_label_encoder[label] for label in labels[:3]]
    print(predictions)

    api = ReedAPI(os.getenv('REED_API'))
    job_des = {}

    for data in predictions:
        jobs = api.search(keywords = data)
        job_des[data] = [] # data is the job title

        for job in jobs[:5]:
            details = job.getJobsDetails()
            job_info = {
                "Job Id" : details.jobId,
                "Job Title" : details.jobTitle,
                "Job Description" : details.jobUrl,
            }
            job_des[data].append(job_info)
    return jsonify({"result" : job_des})


if __name__ == '__main__':
    app.run(host='0.0.0.0',post=5000)
