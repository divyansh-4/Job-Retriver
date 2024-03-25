from flask import Flask, render_template, request, jsonify
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from helper import *

app = Flask(__name__)

# Load the trained model and other necessary objects
with open('./models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('./models/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def parse_pdf(resume_file):
    text = ""
    with pdfplumber.open(resume_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text


# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get input text from the form
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    

    file = request.files['file']

    text = parse_pdf(file)
    # text = request.form['text']

    # Preprocess the input text
    processed_text = preprocess_text(text)

    # Vectorize the processed text
    vectorized_text = tfidf.transform([processed_text])

    # Make prediction using the trained model
    # prediction = model.predict(vectorized_text)
    prediction = model.predict_proba(vectorized_text)[0]
    labels = np.argsort(-prediction)
    # Decode the predicted label
    # predicted_category = encoder.inverse_transform(prediction)[0]
    # prediction = int(prediction[0])
    temp = [[reverse_label_encoder[label], prediction[label]] for label in labels]
    print(temp)
    return jsonify({'category': temp})

if __name__ == '__main__':
    app.run(debug=True)
