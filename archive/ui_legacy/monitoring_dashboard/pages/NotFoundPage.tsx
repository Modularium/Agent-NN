// src/pages/NotFoundPage.tsx
import React from 'react';
import { Link } from 'react-router-dom';
import { AlertTriangle, Home } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const NotFoundPage: React.FC = () => {
  const { isAuthenticated } = useAuth();
  
  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 flex flex-col items-center justify-center p-4">
      <div className="text-center">
        <div className="inline-block p-4 bg-yellow-100 dark:bg-yellow-900/30 rounded-full mb-4">
          <AlertTriangle className="h-16 w-16 text-yellow-600 dark:text-yellow-500" />
        </div>
        
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">404</h1>
        <h2 className="text-2xl font-semibold text-gray-700 dark:text-gray-300 mb-4">Page Not Found</h2>
        
        <p className="text-gray-600 dark:text-gray-400 max-w-md mx-auto mb-8">
          The page you are looking for might have been removed, had its name changed, 
          or is temporarily unavailable.
        </p>
        
        <Link
          to={isAuthenticated ? "/" : "/login"}
          className="inline-flex items-center px-4 py-2 bg-indigo-600 dark:bg-indigo-700 text-white rounded-md hover:bg-indigo-700 dark:hover:bg-indigo-600 transition"
        >
          <Home className="w-5 h-5 mr-2" />
          {isAuthenticated ? "Return to Dashboard" : "Go to Login"}
        </Link>
      </div>
    </div>
  );
};

export default NotFoundPage;
