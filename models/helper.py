import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,WordPunctTokenizer
from string import punctuation
from bs4 import BeautifulSoup
import re
import string
from re import match
from datetime import datetime
from webbrowser import open_new_tab
from requests import get


nltk.download('punkt')
nltk.download('stopwords')

class Job():
    """Job base class"""
    def __init__(self, Jobdict, apikey, apiurl):
        self.apikey = apikey
        self.apiurl = apiurl
        for item in Jobdict:
            if "date" in item.lower():
                # convert dates to datetime objects
                setattr(self, item, datetime.strptime(Jobdict[item], '%d/%m/%Y'))
            else:
                setattr(self, item, Jobdict[item])
    
    def openReedURL(self):
        """Open Reed URL in new tab in default browser. Returns a bool """
        return open_new_tab(self.jobUrl)

    def openExternalURL(self):
        """Open external URL in new tab in default browser Returns a bool"""
        if self.externalUrl == None:
            return False
        else:
            return open_new_tab(self.jobUrl)
    
class SearchJob(Job):
    """For the jobs in search results. Need an easy way to get the details of the job"""
    def getJobsDetails(self):
        response = get(self.apiurl + '/jobs/' + str(self.jobId), auth=(self.apikey, ''))
        return Job(response.json(), self.apikey, self.apiurl)

def parse_pdf(resume_file):
    text = ""
    with pdfplumber.open(resume_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Preprocessing function
def preprocess_text(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    
    text = url_pattern.sub('',text)
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text(separator=' ')

    text = text.lower()

    # Lowercase the text
    contractions = {
        r"\b(can't)\b": "can not",
        r"\b(ain't)\b": "am not",
        r"\b(aren't)\b": "are not",
        r"\b(couldn't)\b": "could not",
        r"\b(didn't)\b": "did not",
        r"\b(doesn't)\b": "does not",
        r"\b(don't)\b": "do not",
        r"\b(hadn't)\b": "had not",
        r"\b(hasn't)\b": "has not",
        r"\b(haven't)\b": "have not",
        r"\b(isn't)\b": "is not",
        r"\b(there's)\b": "there is",
        r"\b(wasn't)\b": "was not",
        r"\b(why's)\b": "why is",
        r"\b(won't)\b": "will not",
        r"\b(wouldn't)\b": "would not",
    }

    # Replace contractions with their expanded forms
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)

    text = text.replace("'s",' is')
    text = text.replace("'m",' am')
    text = text.replace("'re",' are')
    text = text.replace("'ve",' have')
    text = text.replace("'ll",' will')
    text = text.replace("'d",' would')
    text = text.replace("n't",' not')
    text = text.replace("'t",' not')
    # Tokenization
    tokens = word_tokenize(text)
    

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    # stop_words.append("can't")
    tokens = [token for token in tokens if token not in stop_words]

    # Remove punctuations
    tokens = [token for token in tokens if token.isalpha() or token.isalnum()]

    # Remove blank space tokens
    tokens = [token for token in tokens if token.strip() != ' ']

    processed_text = ' '.join(tokens)
    return processed_text

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