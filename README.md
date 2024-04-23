##PPT : https://tome.app/hackathons/job-retriever-model-matching-resumes-with-job-openings-clubqfxfi06sdo65zcl0gl5lz

### Setting up Backend

1. Navigate to backend folder
```bash
cd ./models
```
2. Install dependenceis
```bash
pip install -r requirements.txt
```
3. We are using gemini-pro for optimizing uploaded resume. Get API key for gemini-pro model from https://ai.google.dev/
4. We are using REED api for retreiving job postings. Get API key from https://www.reed.co.uk/developers/jobseeker
5. Create .env file in this directory to integrate API keys. Add API key to this file in following format.
```text
API_KEY='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
REED_API='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
```
6. We finetuned BERT classification model for job matching. Download .pth file from https://drive.google.com/file/d/1fkOr3PZjZdOLGhl3-EzEPu_PtCFEpaAN/view?usp=sharing.
7. Start the server with following command.
```bash
flask run
```

### Setting up Frontend

1. Navigate to frontend folder
```bash
cd ./frontend/resume-optimizer
```
2. Install the dependencies
```bash
npm install
```
3. Run the server
```bash
npm start
```
