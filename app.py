from flask import Flask, render_template, request, jsonify
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from helper import *
from requests import get
from re import match
from job import * 
import requests
import json

app = Flask(__name__)

# Load the trained model and other necessary objects
with open('./models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('./models/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

class ReedAPI():
    """Class to deal with searching and set up
    Usage::
        >>> api = ReedAPI(your_api_key)
        >>> jobs = api.search(keywords='blah blah')
        >>> print(jobs[0].jobId)
        40162354
        >>> print(jobs[0].getJobsDetails().jobDescription)
        (long description with HTML)
    """
    def __init__(self, apikey, apiurl='https://www.reed.co.uk/api/1.0'):
        self.apikey = apikey
        self.apiurl = apiurl
    
    def search(self, employerId=None, employerProfileId=None, keywords=None, locationName=None, distanceFromLocation=None, permanent=None, contract=None, 
               temp=None, partTime=None, fullTime=None, minimumSalary=None, maximumSalary=None, postedByRecruitmentAgency=None, postedByDirectEmployer=None, 
               graduate=None, resultsToTake=50, resultsToSkip=0, pages=None):
        kwargs = locals()
        if pages == 0 or pages == '0' or type(pages) == str:
            raise ValueError('The value of pages must be an int above zero or None (the default, to get all pages)')
        self._validate(kwargs)
        jobs = []

        # Deals with needing to get all the results - the Reed API won't tell you how many results there are until 
        # a query is made
        if pages == None:
            kwargsCopy = kwargs.copy()
            kwargsCopy['resultsToTake'] = 1
            intailRequestURL = self._generateURL(kwargsCopy, self.apiurl + '/search?')
            response = get(intailRequestURL, auth=(self.apikey, ''))
            if response.status_code != 200:
                raise ConnectionError('The Reed API refused the connection. The status returned was ' + str(response.status_code))
            pages = str(response.json()['totalResults'] / resultsToTake + 1).split('.')[0]
            # response.json() doesn't work very well in this case - it errors out
            if "'totalResults': '0'" in response.text or "'totalResults': 0" in response.text or '"totalResults": 0' in response.text:
                raise ValueError('The Reed API returned zero search results')
        
        # Validate resultsToSkip value if one was set
        if kwargs['resultsToSkip'] != 0:
            if resultsToSkip % resultsToTake == 0:
                startingPage = resultsToSkip / resultsToTake
            else:
                raise ValueError('The value of resultsToSkip can not be divided by resultsToTake, so we can not calculate the starting page')
        else:
            startingPage = 0
        
        # Generate the search URL
        searchURL = self._generateURL(kwargs, self.apiurl + '/search?')
        
        # Interate for the number of times requested 
        for i in range(startingPage, int(pages)):
            # Increment resultsToSkip so we don't keep getting the same page
            kwargs['resultsToSkip'] = kwargs['resultsToSkip'] + resultsToTake

            searchURL = self._generateURL(kwargs, self.apiurl + '/search?')
            response = get(searchURL, auth=(self.apikey, ''))

            if response.status_code == 200:
                for job in response.json()['results']:
                    jobs.append(SearchJob(job, self.apikey, self.apiurl))
            else:
                raise ConnectionError('The Reed API refused the connection. The status returned was ' + str(response.status_code))
        return jobs

    def _generateURL(self, kwargs, URLBase):
        count = 0
        for arg in kwargs:
            if arg == 'pages' or kwargs[arg] == 0 or kwargs[arg] == None or arg == 'self':
                continue
            else:
                value = str(kwargs[arg]).replace(' ', '%20').replace('True', 'true').replace('False', 'false').replace('None', 'null')
                if count == 0:
                    URLBase += arg + '=' + value
                else:
                    URLBase += '&' + arg + '=' + value
                count += 1
        return URLBase

    def _validate(self, kwargs):
        for arg in kwargs:
            if match('^(permanent|contract|temp|partTime|fullTime|postedByRecruitmentAgency|postedByDirectEmployer|graduate)$', arg):
                if not match('^(true|false|True|False|None)$', str(kwargs[arg])):
                    raise ValueError('The argument ' + arg + ' needs to be a boolean value or None (the default). It\'s actually ' + str(kwargs[arg]))
        return True

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
    prediction = model.predict_proba(vectorized_text)[0]
    labels = np.argsort(-prediction)
    temp = [[reverse_label_encoder[label], prediction[label]] for label in labels]
    temp=temp[:3]
    temp2=[]
    for t2 in temp:
        temp2.append(t2[0])
    print(temp2)
    api = ReedAPI('1af27b9a-5f60-4961-9643-8e7779c50109')
    jobs_dict = {}

    for keyword in temp2:
        jobs = api.search(keywords=keyword)
        jobs_dict[keyword] = []

        # Get details for the first 5 jobs for the current keyword
        for job in jobs[:5]:
            details = job.getJobsDetails()
            job_info = {
                "Job ID": details.jobId,
                "Job Title": details.jobTitle,
                "Job URL": details.jobUrl
            }
            jobs_dict[keyword].append(job_info)

    # Convert the dictionary to JSON format
    # json_data = json.dumps(jobs_dict, indent=4)
    # return json_data
    return jsonify({'category': jobs_dict})

if __name__ == '__main__':
    app.run(debug=True)
