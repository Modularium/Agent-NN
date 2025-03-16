// src/components/layout/DashboardLayout.tsx
import React, { useState, useEffect } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import Header from './Header';
import Sidebar from './Sidebar';
import Footer from './Footer';
import { useDashboard } from '../../context/DashboardContext';
import LoadingSpinner from '../common/LoadingSpinner';
import Alert from '../common/Alert';
import { useAuth } from '../../context/AuthContext';

const DashboardLayout: React.FC = () => {
  const { refreshData, loading, error } = useDashboard();
  const { isAuthenticated, user } = useAuth();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  
  // Set active tab based on URL path
  useEffect(() => {
    const path = location.pathname.slice(1) || 'system';
    useDashboard().setActiveTab(path);
  }, [location.pathname]);

  // Initial data fetch on mount and setup refresh interval
  useEffect(() => {
    if (isAuthenticated) {
      refreshData();
      
      // Set up auto-refresh
      const interval = setInterval(() => {
        refreshData();
      }, 30000); // Refresh every 30 seconds
      
      return () => clearInterval(interval);
    } else {
      navigate('/login');
    }
  }, [isAuthenticated, refreshData, navigate]);

  // Handle sidebar toggle
  const toggleSidebar = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };

  if (loading && !useDashboard().systemData) {
    return (
      <div className="flex items-center justify-center h-screen">
        <LoadingSpinner size="lg" text="Loading dashboard data..." />
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-gray-100 dark:bg-gray-900">
      <Header />
      
      <div className="flex flex-1 overflow-hidden">
        <Sidebar 
          collapsed={sidebarCollapsed} 
          toggleCollapse={toggleSidebar} 
        />
        
        <main className="flex-1 overflow-auto p-6 bg-gray-100 dark:bg-gray-900">
          {error && (
            <Alert 
              type="error" 
              title="Error loading data" 
              message={error.message}
              className="mb-4"
            />
          )}
          
          {/* This is where the route's component will be rendered */}
          <Outlet />
        </main>
      </div>
      
      <Footer />
    </div>
  );
};

export default DashboardLayout;
