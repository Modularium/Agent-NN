// MainContent.tsx - Main content container for the dashboard
import React from 'react';
import { useDashboard } from '../../context/DashboardContext';
import SystemOverviewPanel from '../panels/SystemOverviewPanel';
import AgentsPanel from '../panels/AgentsPanel';
import ModelsPanel from '../panels/ModelsPanel';
import KnowledgePanel from '../panels/KnowledgePanel';
import MonitoringPanel from '../panels/MonitoringPanel';
import SecurityPanel from '../panels/SecurityPanel';
import TestingPanel from '../panels/TestingPanel';
import SettingsPanel from '../panels/SettingsPanel';
import LogsPanel from '../panels/LogsPanel';
import DocsPanel from '../panels/DocsPanel';
import LoadingSpinner from '../common/LoadingSpinner';
import Alert from '../common/Alert';

const MainContent: React.FC = () => {
  const { activeTab, loading, error } = useDashboard();
  
  if (loading) {
    return (
      <div className="flex items-center justify-center h-full p-6">
        <LoadingSpinner fullPage text="Loading dashboard data..." />
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

export default MainContent;
