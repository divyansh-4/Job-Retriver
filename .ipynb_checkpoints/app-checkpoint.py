from flask import Flask, render_template, request, jsonify
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np


app = Flask(__name__)

# Load the trained model and other necessary objects
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# with open('encoder.pkl', 'rb') as f:
#     encoder = pickle.load(f)
label_encoder = {'Data Science': 0,
 'HR': 1,
 'Advocate': 2,
 'Arts': 3,
 'Web Designing': 4,
 'Mechanical Engineer': 5,
 'Sales': 6,
 'Health and fitness': 7,
 'Civil Engineer': 8,
 'Java Developer': 9,
 'Business Analyst': 10,
 'SAP Developer': 11,
 'Automation Testing': 12,
 'Electrical Engineering': 13,
 'Operations Manager': 14,
 'Python Developer': 15,
 'DevOps Engineer': 16,
 'Network Security Engineer': 17,
 'PMO': 18,
 'Database': 19,
 'Hadoop': 20,
 'ETL Developer': 21,
 'DotNet Developer': 22,
 'Blockchain': 23,
 'Testing': 24}

reverse_label_encoder = {v: k for k, v in label_encoder.items()}

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get input text from the form
    text = request.form['text']

    # Preprocess the input text
    processed_text = preprocess_text(text)

    # Vectorize the processed text
    vectorized_text = tfidf.transform([processed_text])

    # Make prediction using the trained model
    prediction = model.predict(vectorized_text)

    # Decode the predicted label
    # predicted_category = encoder.inverse_transform(prediction)[0]
    prediction = int(prediction[0])
    return jsonify({'category': reverse_label_encoder[prediction]})

# Preprocessing function
def preprocess_text(text):
    # Implement your preprocessing logic here
    # Example: lowercase, remove punctuation, etc.
    return text.lower()

if __name__ == '__main__':
    app.run(debug=True)
