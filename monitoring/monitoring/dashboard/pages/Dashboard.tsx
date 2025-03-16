// monitoring/dashboard/pages/Dashboard.tsx
import React, { useState, useEffect } from 'react';
import Header from '../components/layout/Header';
import Sidebar from '../components/layout/Sidebar';
import Footer from '../components/layout/Footer';
import { useAuth } from '../context/AuthContext';
import { useDashboard } from '../context/DashboardContext';
import LoadingSpinner from '../components/common/LoadingSpinner';
import Alert from '../components/common/Alert';

// Import all panel components
import SystemOverviewPanel from '../components/panels/SystemOverviewPanel';
import AgentsPanel from '../components/panels/AgentsPanel';
import ModelsPanel from '../components/panels/ModelsPanel';
import KnowledgePanel from '../components/panels/KnowledgePanel';
import MonitoringPanel from '../components/panels/MonitoringPanel';
import SecurityPanel from '../components/panels/SecurityPanel';
import TestingPanel from '../components/panels/TestingPanel';
import SettingsPanel from '../components/panels/SettingsPanel';
import LogsPanel from '../components/panels/LogsPanel';
import DocsPanel from '../components/panels/DocsPanel';

// Login form component
const LoginForm: React.FC = () => {
  const { login, loading, error } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await login(username, password);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100 dark:bg-gray-900 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h1 className="text-center text-3xl font-extrabold text-indigo-600 dark:text-indigo-400">Agent-NN Dashboard</h1>
          <h2 className="mt-6 text-center text-2xl font-bold text-gray-900 dark:text-white">Sign in to your account</h2>
        </div>
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="rounded-md shadow-sm -space-y-px">
            <div>
              <label htmlFor="username" className="sr-only">Username</label>
              <input
                id="username"
                name="username"
                type="text"
                required
                className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white placeholder-gray-500 text-gray-900 rounded-t-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm"
                placeholder="Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>
            <div>
              <label htmlFor="password" className="sr-only">Password</label>
              <input
                id="password"
                name="password"
                type="password"
                required
                className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white placeholder-gray-500 text-gray-900 rounded-b-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
          </div>

          {error && (
            <Alert 
              type="error" 
              message={error} 
            />
          )}

          <div>
            <button
              type="submit"
              disabled={loading}
              className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-70 transition"
            >
              {loading ? 'Signing in...' : 'Sign in'}
            </button>
          </div>
        </form>
        
        <div className="text-center text-xs text-gray-500 dark:text-gray-400 mt-4">
          <p>Demo credentials: username: admin, password: password</p>
        </div>
      </div>
    </div>
  );
};

// Main content component
const MainContent: React.FC = () => {
  const { activeTab, loading, error, systemData } = useDashboard();
  
  if (loading) {
    return (
      <div className="flex items-center justify-center h-full p-6">
        <LoadingSpinner size="lg" text="Loading dashboard data..." />
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="flex items-center justify-center h-full p-6">
        <Alert 
          type="error" 
          title="Failed to load dashboard data" 
          message={error.message || 'An unexpected error occurred while loading the dashboard.'}
        />
      </div>
    );
  }

  if (!systemData && activeTab !== 'docs') {
    return (
      <div className="flex items-center justify-center h-full p-6">
        <Alert 
          type="warning" 
          title="No data available" 
          message="The system is not returning any data. Please check your connection or try again later."
        />
      </div>
    );
  }
  
  return (
    <main className="flex-1 overflow-auto p-6 bg-gray-100 dark:bg-gray-900">
      {activeTab === 'system' && <SystemOverviewPanel />}
      {activeTab === 'agents' && <AgentsPanel />}
      {activeTab === 'models' && <ModelsPanel />}
      {activeTab === 'knowledge' && <KnowledgePanel />}
      {activeTab === 'monitoring' && <MonitoringPanel />}
      {activeTab === 'security' && <SecurityPanel />}
      {activeTab === 'testing' && <TestingPanel />}
      {activeTab === 'settings' && <SettingsPanel />}
      {activeTab === 'logs' && <LogsPanel />}
      {activeTab === 'docs' && <DocsPanel />}
    </main>
  );
};

// Main Dashboard component
const Dashboard: React.FC = () => {
  const { isAuthenticated, loading: authLoading } = useAuth();
  const { refreshData } = useDashboard();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Initial data fetch on mount
  useEffect(() => {
    if (isAuthenticated) {
      refreshData();
      
      // Set up auto-refresh (optional)
      const interval = setInterval(() => {
        refreshData();
      }, 30000); // Refresh every 30 seconds
      
      return () => clearInterval(interval);
    }
  }, [isAuthenticated, refreshData]);

  if (authLoading) {
    return <LoadingSpinner fullPage text="Loading..." />;
  }

  if (!isAuthenticated) {
    return <LoginForm />;
  }

  return (
    <div className="flex flex-col h-screen bg-gray-100 dark:bg-gray-900">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar collapsed={sidebarCollapsed} toggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)} />
        <MainContent />
      </div>
      <Footer />
    </div>
  );
};

export default Dashboard;
