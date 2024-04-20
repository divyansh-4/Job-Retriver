import React from 'react';
import { Link } from 'react-router-dom';

const Home = () => {
    return (
        <div className="flex flex-col items-center justify-center h-screen">
            <div>
                <p className="text-red-500 dark:text-green-400">We are offering following services for now!</p>
            </div>
            <div className="mt-3">
                <Link to='/job-finder'>
                    <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mr-4">Find Jobs</button>
                </Link>
                <Link to='/optimise'>
                    <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mr-4">Optimize resume</button>
                </Link>
                <Link to='/similarity'>
                    <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Job Matcher</button>
                </Link>
            </div>
        </div>
    )
}

export { Home }
