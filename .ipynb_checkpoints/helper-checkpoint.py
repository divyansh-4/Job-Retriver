import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,WordPunctTokenizer
from string import punctuation
from bs4 import BeautifulSoup
import re
import string

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