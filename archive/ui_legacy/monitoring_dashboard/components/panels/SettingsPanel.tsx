// monitoring/dashboard/components/panels/SettingsPanel.tsx
import React, { useState } from 'react';
import { Save, RefreshCw, AlertTriangle, Shield, Bell, Server, Database, Clock, User, Mail, Globe } from 'lucide-react';
import Card from '../common/Card';
import Alert from '../common/Alert';
import TabView from '../common/TabView';
import { useTheme } from '../../context/ThemeContext';
import { useDashboard } from '../../context/DashboardContext';

const SettingsPanel: React.FC = () => {
  const { themeMode, setThemeMode } = useTheme();
  const { refreshData } = useDashboard();
  
  // Settings states
  const [generalSettings, setGeneralSettings] = useState({
    systemName: 'Agent-NN System',
    timezone: 'UTC',
    language: 'en',
    dateFormat: 'YYYY-MM-DD',
    timeFormat: '24h'
  });
  
  const [notificationSettings, setNotificationSettings] = useState({
    emailNotifications: true,
    emailAddress: 'admin@example.com',
    slackWebhook: 'https://hooks.slack.com/services/xxx/yyy/zzz',
    slackEnabled: true,
    desktopNotifications: true,
    criticalAlertsOnly: false
  });
  
  const [securitySettings, setSecuritySettings] = useState({
    sessionTimeout: 30,
    maxLoginAttempts: 5,
    requireMFA: false,
    passwordExpiry: 90,
    apiTokenExpiry: 30
  });
  
  const [apiSettings, setApiSettings] = useState({
    maxRequestsPerMinute: 120,
    maxConcurrentRequests: 20,
    requestTimeout: 60,
    apiLogging: true,
    enableRateLimiting: true
  });
  
  const [backupSettings, setBackupSettings] = useState({
    automaticBackups: true,
    backupFrequency: 'daily',
    retentionPeriod: 30,
    backupTime: '01:00',
    includeLogs: true,
    includeMetrics: false
  });
  
  // Form state
  const [isSaving, setIsSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<{type: 'success' | 'error'; message: string} | null>(null);
  
  // Handle saving settings
  const handleSaveSettings = async () => {
    setIsSaving(true);
    setSaveMessage(null);
    
    try {
      // In a real implementation, this would call an API to save settings
      await new Promise(resolve => setTimeout(resolve, 1000));
      setSaveMessage({
        type: 'success',
        message: 'Settings saved successfully'
      });
    } catch (error) {
      setSaveMessage({
        type: 'error',
        message: 'Failed to save settings. Please try again.'
      });
    } finally {
      setIsSaving(false);
    }
  };
  
  // Handle reset to defaults
  const handleResetSettings = () => {
    // In a real implementation, this would reset to default values from backend
    if (window.confirm('Are you sure you want to reset all settings to default values? This cannot be undone.')) {
      // Reset logic would go here
      setSaveMessage({
        type: 'success',
        message: 'Settings reset to default values'
      });
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">System Settings</h2>
        <div className="flex space-x-3">
          <button 
            className="px-4 py-2 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600 flex items-center transition"
            onClick={handleResetSettings}
          >
            <RefreshCw size={16} className="mr-2" />
            Reset to Defaults
          </button>
          <button 
            className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 flex items-center transition"
            onClick={handleSaveSettings}
            disabled={isSaving}
          >
            <Save size={16} className="mr-2" />
            {isSaving ? 'Saving...' : 'Save Settings'}
          </button>
        </div>
      </div>

      {/* Save message alert */}
      {saveMessage && (
        <Alert 
          type={saveMessage.type} 
          message={saveMessage.message} 
          onClose={() => setSaveMessage(null)}
          closable
        />
      )}

      {/* Settings Tabs */}
      <TabView
        tabs={[
          {
            id: 'general',
            label: 'General',
            icon: <Server size={16} className="mr-2" />,
            content: (
              <Card className="mt-4" noPadding={false}>
                <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">System Name</label>
                    <input 
                      type="text" 
                      className="form-input rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white" 
                      value={generalSettings.systemName}
                      onChange={(e) => setGeneralSettings({...generalSettings, systemName: e.target.value})}
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Timezone</label>
                    <select 
                      className="form-select rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
                      value={generalSettings.timezone}
                      onChange={(e) => setGeneralSettings({...generalSettings, timezone: e.target.value})}
                    >
                      <option value="UTC">UTC</option>
                      <option value="America/New_York">Eastern Time (ET)</option>
                      <option value="America/Chicago">Central Time (CT)</option>
                      <option value="America/Denver">Mountain Time (MT)</option>
                      <option value="America/Los_Angeles">Pacific Time (PT)</option>
                      <option value="Europe/London">London (GMT)</option>
                      <option value="Europe/Paris">Paris (CET)</option>
                      <option value="Asia/Tokyo">Tokyo (JST)</option>
                    </select>
                  </div>
                </div>

                <div className="mt-6 space-y-4">
                  <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">Data Refresh Settings</h3>
                  
                  <div className="flex items-center justify-between">
                    <label className="text-sm text-gray-700 dark:text-gray-300">Auto-refresh dashboard data</label>
                    <div className="relative inline-block w-12 h-6 border rounded-full border-gray-300 dark:border-gray-700">
                      <input type="checkbox" className="sr-only" defaultChecked />
                      <span className="block absolute left-1 top-1 bg-indigo-600 dark:bg-indigo-500 w-4 h-4 rounded-full transition-transform transform translate-x-6"></span>
                    </div>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Refresh Interval</label>
                    <select className="form-select rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white">
                      <option value="10000">10 seconds</option>
                      <option value="30000" selected>30 seconds</option>
                      <option value="60000">1 minute</option>
                      <option value="300000">5 minutes</option>
                      <option value="600000">10 minutes</option>
                    </select>
                  </div>
                </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Language</label>
                    <select 
                      className="form-select rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
                      value={generalSettings.language}
                      onChange={(e) => setGeneralSettings({...generalSettings, language: e.target.value})}
                    >
                      <option value="en">English</option>
                      <option value="de">Deutsch</option>
                      <option value="fr">Français</option>
                      <option value="es">Español</option>
                      <option value="ja">日本語</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Theme</label>
                    <select 
                      className="form-select rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
                      value={themeMode}
                      onChange={(e) => setThemeMode(e.target.value as 'light' | 'dark')}
                    >
                      <option value="light">Light</option>
                      <option value="dark">Dark</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Date Format</label>
                    <select 
                      className="form-select rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
                      value={generalSettings.dateFormat}
                      onChange={(e) => setGeneralSettings({...generalSettings, dateFormat: e.target.value})}
                    >
                      <option value="YYYY-MM-DD">YYYY-MM-DD</option>
                      <option value="MM/DD/YYYY">MM/DD/YYYY</option>
                      <option value="DD/MM/YYYY">DD/MM/YYYY</option>
                      <option value="DD.MM.YYYY">DD.MM.YYYY</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Time Format</label>
                    <select 
                      className="form-select rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
                      value={generalSettings.timeFormat}
                      onChange={(e) => setGeneralSettings({...generalSettings, timeFormat: e.target.value})}
                    >
                      <option value="24h">24 hour (13:30)</option>
                      <option value="12h">12 hour (1:30 PM)
