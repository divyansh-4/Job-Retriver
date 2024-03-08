from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
import pdfplumber
from io import BytesIO

app = Flask(__name__)

# Load the trained model and other necessary objects
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Load label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

reverse_label_encoder = {v: k for k, v in label_encoder.items()}

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is a PDF
    if file.filename.endswith('.pdf'):
        # Extract text from PDF
        text = extract_text_from_pdf(file)
    else:
        return jsonify({'error': 'Unsupported file format'})

    # Preprocess the extracted text
    processed_text = preprocess_text(text)

    # Vectorize the processed text
    vectorized_text = tfidf.transform([processed_text])

    # Make prediction using the trained model
    prediction = model.predict(vectorized_text)

    # Decode the predicted label
    predicted_category = reverse_label_encoder[prediction[0]]

    return jsonify({'category': predicted_category})

# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        pdf_data = BytesIO(file.read())
        with pdfplumber.open(pdf_data) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

# Preprocessing function
def preprocess_text(text):
    # Implement your preprocessing logic here
    # Example: lowercase, remove punctuation, etc.
    return text.lower()

if __name__ == '__main__':
    app.run(debug=True)
