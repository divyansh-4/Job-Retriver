import React from 'react';

const About = () => {
    return (
        <div className="container mx-auto my-8">
            <div className="max-w-4xl mx-auto px-4">
                <h1 className="text-3xl font-bold mb-6">About Our Services</h1>
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
                    <h2 className="text-xl font-semibold mb-4 dark:text-white">Our Mission</h2>
                    <p className="text-gray-700 dark:text-gray-300">
                        At Resoptim Services, we're committed to revolutionizing the job search process. We understand the challenges individuals face when looking for new opportunities or optimizing their resumes, and we're here to provide innovative solutions.
                    </p>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mt-6">
                    <h2 className="text-xl font-semibold mb-4 dark:text-white">Our Services</h2>
                    <ul className="list-disc list-inside text-gray-700 dark:text-gray-300">
                        <li><strong>BERT-Based Job Classification:</strong> Utilizing state-of-the-art BERT technology, we match job seekers with relevant job listings based on their uploaded resumes.</li>
                        <li><strong>Gemini 2 Resume Optimization:</strong> We leverage Gemini 2 to identify and address shortcomings in resumes, ensuring they stand out to potential employers.</li>
                        <li><strong>Resume-to-Job Matching:</strong> Our platform analyzes resumes and suggests suitable job opportunities, helping individuals find their ideal career paths.</li>
                    </ul>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mt-6">
                    <h2 className="text-xl font-semibold mb-4 dark:text-white">Our Team</h2>
                    <p className="text-gray-700 dark:text-gray-300">
                        Meet the dedicated team behind Resoptim Services, comprised of talented individuals passionate about improving the job search experience. With expertise in machine learning, natural language processing, and user experience design, we're committed to delivering top-notch solutions to our users.
                    </p>
                </div>
            </div>
        </div>
    );
}

export {About}
