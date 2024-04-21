import React, { useState } from 'react';
import axios from 'axios';
import {result} from '../temp-data'

const FileUpload = () => {

    const [resume, setResume] = useState(null);
    const [responseData, setResponseData] = useState(null);

    const handleFileChange = (e) => {
        setResume(e.target.files[0]);
    }

    const handleUpload = async () => {
        if (resume) {
            const formData = new FormData();
            formData.append('resume', resume);

            try {
                // const res = await axios.post('http://localhost:5000/predict', formData, {});
                // setResponseData(res.data);
                // console.log(res.data);
                // console.log('File uploaded successfully');

                // Temporary response data
                setResponseData(result);
            } catch (error) {
                console.log(error);
            }
        } else {
            console.log('No file selected');
        }
    }

    return (
      <div className="container mx-auto my-8">
          {!responseData && (
            <div className="max-w-lg mx-auto p-6 bg-gray-100 dark:bg-gray-700 rounded-lg shadow-md">
                <label htmlFor="dropzone-file" className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-600 transition duration-300">
                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                        <svg className="w-8 h-8 mb-4 text-gray-500 dark:text-gray-400" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
                            <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"/>
                        </svg>
                        <p className="mb-2 text-sm text-gray-500 dark:text-gray-400"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                        <p className="text-xs text-gray-500 dark:text-gray-400">PDF</p>
                    </div>
                    <input id="dropzone-file" type="file" onChange={handleFileChange} accept='.pdf'/>
                </label>
                <button onClick={handleUpload} className="mt-4 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Upload</button>
            </div>  
          )}
        
  

        {responseData && (
            <div className="mt-8 flex justify-center items-center">
            <div>
                {Object.keys(responseData.result).map((key) => (
                    <div key={key} className="mb-8">
                        <h2 className="text-lg font-semibold mb-4 dark:text-blue-400">{key}</h2>
                        {responseData.result[key].map((value) => (
                            <div key={value['Job Id']} className="mb-4">
                                <h3 className="text-base font-semibold mb-2">{value['Job Title']}</h3>
                                <p className="text-sm mb-2">{value['Job Description']}</p>
                                <a href={value['Job Description']} className="text-blue-600 hover:underline">View Job Description</a>
                            </div>
                        ))}
                    </div>
                ))}
            </div>
        </div>
        
        )}
        </div>
    );
}

export { FileUpload }
