import {BrowserRouter,Routes,Route} from 'react-router-dom'
import {Home} from './components/Home'
import { FileUpload } from './components/fileUpload';
import { Optimise } from './components/optimiser';
import './index.css';
import { Similarity } from './components/jobSimilarity';
import {About} from './components/about'


function App() {

  return (
    <div className="App">
    <nav class="bg-white border-gray-200 dark:bg-gray-900">
      <div class="max-w-screen-xl flex flex-wrap items-center justify-between mx-auto p-4">
        <h1 class=' font-extrabold leading-none tracking-tight text-gray-900 md:text-3xl lg:text-2xl dark:text-white'>Resoptim</h1>
        <div class="hidden w-full md:block md:w-auto" id="navbar-default">
          <ul class="font-medium flex flex-col p-4 md:p-0 mt-4 border border-gray-100 rounded-lg bg-gray-50 md:flex-row md:space-x-8 rtl:space-x-reverse md:mt-0 md:border-0 md:bg-white dark:bg-gray-800 md:dark:bg-gray-900 dark:border-gray-700">
            <li>
              <a href="/" class="block py-2 px-3 text-white bg-blue-700 rounded md:bg-transparent md:text-blue-700 md:p-0 dark:text-white md:dark:text-blue-500" aria-current="page">Home</a>
            </li>
            <li>
              <a href="/about" class="block py-2 px-3 text-gray-900 rounded hover:bg-gray-100 md:hover:bg-transparent md:border-0 md:hover:text-blue-700 md:p-0 dark:text-white md:dark:hover:text-blue-500 dark:hover:bg-gray-700 dark:hover:text-white md:dark:hover:bg-transparent">About</a>
            </li>
            <li>
              <a href="#" class="block py-2 px-3 text-gray-900 rounded hover:bg-gray-100 md:hover:bg-transparent md:border-0 md:hover:text-blue-700 md:p-0 dark:text-white md:dark:hover:text-blue-500 dark:hover:bg-gray-700 dark:hover:text-white md:dark:hover:bg-transparent">Contact</a>
            </li>
          </ul>
        </div>
      </div>
    </nav>


      <BrowserRouter>
        <Routes>
          <Route path='/' element={<Home/>}/>
          <Route path='/job-finder' element={<FileUpload/>}/>
          <Route path='/optimise' element={<Optimise/>}/>
          <Route path='/similarity' element={<Similarity/>}/>
          <Route path='/about' element={<About/>}/>
        </Routes>
      </BrowserRouter>
      {/* <FileUpload/> */}
    </div>
  );
}

export default App;
