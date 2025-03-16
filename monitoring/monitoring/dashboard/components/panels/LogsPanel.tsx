// monitoring/dashboard/components/panels/LogsPanel.tsx
import React, { useState } from 'react';
import { Search, Download, Trash2, Filter, Bell, AlertCircle, Clock, RefreshCw } from 'lucide-react';
import { useDashboard } from '../../context/DashboardContext';
import Card from '../common/Card';
import StatusBadge from '../common/StatusBadge';

const LogsPanel: React.FC = () => {
  const { logs } = useDashboard();
  const [levelFilter, setLevelFilter] = useState<string>('all');
  const [timeRange, setTimeRange] = useState<string>('24h');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [page, setPage] = useState<number>(1);
  const [activeTab, setActiveTab] = useState<'logs' | 'alerts' | 'config'>('logs');
  
  // Mock alert configurations
  const alertConfigs = [
    {
      id: 'alert-1',
      name: 'CPU Usage Alert',
      condition: 'CPU usage exceeds 80% for 5 minutes',
      channels: ['Email', 'Dashboard'],
      enabled: true,
      severity: 'critical'
    },
    {
      id: 'alert-2',
      name: 'Memory Usage Alert',
      condition: 'Memory usage exceeds 90% for 5 minutes',
      channels: ['Email', 'Dashboard', 'Slack'],
      enabled: true,
      severity: 'critical'
    },
    {
      id: 'alert-3',
      name: 'Agent Failure Alert',
      condition: 'Agent success rate drops below 75%',
      channels: ['Email', 'Dashboard'],
      enabled: true,
      severity: 'warning'
    },
    {
      id: 'alert-4',
      name: 'Task Queue Alert',
      condition: 'Task queue size exceeds 50',
      channels: ['Dashboard'],
      enabled: false,
      severity: 'warning'
    }
  ];
  
  // Filter logs by level
  const filteredLogs = logs.filter(log => {
    // Apply level filter
    if (levelFilter !== 'all' && log.level !== levelFilter) {
      return false;
    }
    
    // Apply search filter
    if (searchQuery && !log.message.toLowerCase().includes(searchQuery.toLowerCase())) {
      return false;
    }
    
    // Apply time filter (this is simplified, in a real app you'd use proper date filtering)
    // Here we just return true since we don't have a way to properly filter by time in our mock data
    return true;
  });
  
  // Paginate logs
  const logsPerPage = 10;
  const paginatedLogs = filteredLogs.slice((page - 1) * logsPerPage, page * logsPerPage);
  const totalPages = Math.ceil(filteredLogs.length / logsPerPage);
  
  // Get severity class for log level
  const getLevelClass = (level: string): string => {
    switch (level) {
      case 'ERROR':
        return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300';
      case 'WARNING':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300';
      case 'INFO':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300';
      case 'DEBUG':
        return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300';
    }
  };
  
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">Logs & Alerts</h2>
        <div className="flex items-center space-x-3">
          <select 
            className="form-select rounded-md border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white" 
            value={levelFilter}
            onChange={(e) => setLevelFilter(e.target.value)}
          >
            <option value="all">All Levels</option>
            <option value="ERROR">ERROR</option>
            <option value="WARNING">WARNING</option>
            <option value="INFO">INFO</option>
            <option value="DEBUG">DEBUG</option>
          </select>
          <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 flex items-center transition">
            <Download size={18} className="mr-2" />
            Export Logs
          </button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-gray-200 dark:border-gray-700">
        <button
          className={`px-4 py-2 font-medium ${
            activeTab === 'logs'
              ? 'border-b-2 border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('logs')}
        >
          System Logs
        </button>
        <button
          className={`px-4 py-2 font-medium ${
            activeTab === 'alerts'
              ? 'border-b-2 border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('alerts')}
        >
          Recent Alerts
        </button>
        <button
          className={`px-4 py-2 font-medium ${
            activeTab === 'config'
              ? 'border-b-2 border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('config')}
        >
          Alert Configuration
        </button>
      </div>

      {/* Logs Tab */}
      {activeTab === 'logs' && (
        <Card>
          <div className="flex justify-between items-center mb-4">
            <div className="flex items-center space-x-2">
              <div className="relative">
                <input
                  type="text"
                  className="pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white rounded-md"
                  placeholder="Search logs..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search size={18} className="text-gray-400" />
                </div>
              </div>
              <select 
                className="form-select rounded-md border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value)}
              >
                <option value="1h">Last 1 hour</option>
                <option value="6h">Last 6 hours</option>
                <option value="24h">Last 24 hours</option>
                <option value="7d">Last 7 days</option>
                <option value="30d">Last 30 days</option>
              </select>
              <button className="p-2 border border-gray-300 dark:border-gray-700 rounded-md text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300">
                <Filter size={18} />
              </button>
              <button className="p-2 border border-gray-300 dark:border-gray-700 rounded-md text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300">
                <RefreshCw size={18} />
              </button>
            </div>
            <button className="px-3 py-1 bg-red-100 text-red-700 rounded-md hover:bg-red-200 dark:bg-red-900/30 dark:text-red-400 dark:hover:bg-red-900/50 flex items-center transition">
              <Trash2 size={16} className="mr-1" />
              Clear Logs
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead>
                <tr>
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Timestamp</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Level</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Message</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {paginatedLogs.map((log, index) => (
                  <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                    <td className="px-3 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {new Date(log.timestamp).toLocaleString()}
                    </td>
                    <td className="px-3 py-4 whitespace-nowrap text-sm">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getLevelClass(log.level)}`}>
                        {log.level}
                      </span>
                    </td>
                    <td className="px-3 py-4 text-sm text-gray-900 dark:text-white">
                      {log.message}
                    </td>
                    <td className="px-3 py-4 whitespace-nowrap text-sm">
                      <button className="px-2 py-1 bg-indigo-100 text-indigo-700 rounded text-xs dark:bg-indigo-900/30 dark:text-indigo-400">
                        Details
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex justify-between items-center mt-4">
            <div className="text-sm text-gray-500 dark:text-gray-400">
              Showing {((page - 1) * logsPerPage) + 1} to {Math.min(page * logsPerPage, filteredLogs.length)} of {filteredLogs.length} logs
            </div>
            <div className="flex space-x-2">
              <button 
                className={`px-3 py-1 rounded-md text-sm ${page === 1 ? 'bg-gray-100 text-gray-400 cursor-not-allowed dark:bg-gray-800 dark:text-gray-600' : 'bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700'}`}
                onClick={() => setPage(page - 1)}
                disabled={page === 1}
              >
                Previous
              </button>
              {[...Array(Math.min(5, totalPages))].map((_, i) => {
                // Show pages around current page
                let pageNum = i + 1;
                if (totalPages > 5 && page > 3) {
                  pageNum = page - 3 + i;
                }
                if (pageNum > totalPages) return null;
                
                return (
                  <button
                    key={i}
                    className={`px-3 py-1 rounded-md text-sm ${pageNum === page ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-400' : 'bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700'}`}
                    onClick={() => setPage(pageNum)}
                  >
                    {pageNum}
                  </button>
                );
              })}
              <button 
                className={`px-3 py-1 rounded-md text-sm ${page === totalPages ? 'bg-gray-100 text-gray-400 cursor-not-allowed dark:bg-gray-800 dark:text-gray-600' : 'bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700'}`}
                onClick={() => setPage(page + 1)}
                disabled={page === totalPages}
              >
                Next
              </button>
            </div>
          </div>
        </Card>
      )}

      {/* Alerts Tab */}
      {activeTab === 'alerts' && (
        <Card>
          <div className="space-y-4">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg border border-yellow-200 dark:border-yellow-800">
              <div className="flex items-start">
                <AlertCircle className="text-yellow-500 w-5 h-5 mt-0.5 mr-3 flex-shrink-0" />
                <div>
                  <h3 className="font-medium text-yellow-800 dark:text-yellow-300">High Memory Usage</h3>
                  <p className="text-sm text-yellow-700 dark:text-yellow-400 mt-1">
                    Memory usage has reached 78%, which exceeds the warning threshold of 75%.
                  </p>
                  <div className="flex items-center mt-2 text-xs text-yellow-600 dark:text-yellow-500">
                    <Clock size={14} className="mr-1" />
                    <span>2 minutes ago</span>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
              <div className="flex items-start">
                <Bell className="text-blue-500 w-5 h-5 mt-0.5 mr-3 flex-shrink-0" />
                <div>
                  <h3 className="font-medium text-gray-800 dark:text-gray-300">Daily System Backup Completed</h3>
                  <p className="text-sm text-gray-700 dark:text-gray-400 mt-1">
                    The daily system backup was successfully completed.
                  </p>
                  <div className="flex items-center mt-2 text-xs text-gray-600 dark:text-gray-500">
                    <Clock size={14} className="mr-1" />
                    <span>2 hours ago</span>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border border-red-200 dark:border-red-800">
              <div className="flex items-start">
                <AlertCircle className="text-red-500 w-5 h-5 mt-0.5 mr-3 flex-shrink-0" />
                <div>
                  <h3 className="font-medium text-red-800 dark:text-red-300">Failed to Load Model</h3>
                  <p className="text-sm text-red-700 dark:text-red-400 mt-1">
                    Failed to load model "mistral-7b" - CUDA out of memory
                  </p>
                  <div className="flex items-center mt-2 text-xs text-red-600 dark:text-red-500">
                    <Clock size={14} className="mr-1" />
                    <span>5 minutes ago</span>
                  </div>
                </div>
              </div>
            </div>
            
            <h3 className="font-bold mt-6 mb-3 text-gray-900 dark:text-white">Yesterday</h3>
            
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-800">
              <div className="flex items-start">
                <Bell className="text-blue-500 w-5 h-5 mt-0.5 mr-3 flex-shrink-0" />
                <div>
                  <h3 className="font-medium text-blue-800 dark:text-blue-300">System Update Available</h3>
                  <p className="text-sm text-blue-700 dark:text-blue-400 mt-1">
                    A new system update (v2.1.0) is available. Click to review and install.
                  </p>
                  <div className="flex items-center mt-2 text-xs text-blue-600 dark:text-blue-500">
                    <Clock size={14} className="mr-1" />
                    <span>Yesterday at 14:30</span>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border border-red-200 dark:border-red-800">
              <div className="flex items-start">
                <AlertCircle className="text-red-500 w-5 h-5 mt-0.5 mr-3 flex-shrink-0" />
                <div>
                  <h3 className="font-medium text-red-800 dark:text-red-300">Connection Error</h3>
                  <p className="text-sm text-red-700 dark:text-red-400 mt-1">
                    Failed to connect to the OpenAI API. Check your internet connection and API key.
                  </p>
                  <div className="flex items-center mt-2 text-xs text-red-600 dark:text-red-500">
                    <Clock size={14} className="mr-1" />
                    <span>Yesterday at 10:15</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Alert Configuration Tab */}
      {activeTab === 'config' && (
        <>
          <Card title="Alert Configuration">
            <div className="mb-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="font-semibold text-gray-900 dark:text-white">System Alerts</h3>
                <button className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded text-sm dark:bg-indigo-900/30 dark:text-indigo-400 hover:bg-indigo-200 dark:hover:bg-indigo-900/50 transition">
                  Add New Alert
                </button>
              </div>
              
              <div className="space-y-4">
                {alertConfigs.map(alert => (
                  <div key={alert.id} className="border rounded-lg p-4 dark:border-gray-700">
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="flex items-center">
                          <h4 className="font-medium text-gray-900 dark:text-white">{alert.name}</h4>
                          <span className={`ml-2 px-2 py-0.5 rounded-full text-xs font-medium ${
                            alert.severity === 'critical' 
                              ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300' 
                              : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300'
                          }`}>
                            {alert.severity}
                          </span>
                        </div>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{alert.condition}</p>
                        <div className="flex items-center mt-2">
                          <span className="text-xs text-gray-500 dark:text-gray-400 mr-2">Notify via:</span>
                          {alert.channels.map((channel, index) => (
                            <span key={index} className="mr-2 px-2 py-0.5 bg-gray-100 text-gray-700 rounded text-xs dark:bg-gray-800 dark:text-gray-300">
                              {channel}
                            </span>
                          ))}
                        </div>
                      </div>
                      <div className="flex items-center">
                        <div className="relative inline-block w-12 h-6 border rounded-full mr-3 dark:border-gray-700">
                          <input type="checkbox" className="sr-only" defaultChecked={alert.enabled} />
                          <span 
                            className={`block absolute left-1 top-1 w-4 h-4 rounded-full transition-transform ${
                              alert.enabled 
                                ? 'bg-indigo-600 dark:bg-indigo-500 transform translate-x-6' 
                                : 'bg-gray-300 dark:bg-gray-600'
                            }`}
                          ></span>
                        </div>
                        <button className="p-1 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300">
                          <Edit size={16} />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="mb-6">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-4">Notification Channels</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="border rounded-lg p-4 dark:border-gray-700">
                  <div className="flex justify-between items-center mb-2">
                    <h4 className="font-medium text-gray-900 dark:text-white">Email Notifications</h4>
                    <div className="relative inline-block w-12 h-6 border rounded-full dark:border-gray-700">
                      <input type="checkbox" className="sr-only" defaultChecked />
                      <span className="block absolute left-1 top-1 bg-indigo-600 dark:bg-indigo-500 w-4 h-4 rounded-full transition-transform transform translate-x-6"></span>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    Receive system alerts via email
                  </p>
                  <div className="flex items-center">
                    <input 
                      type="email" 
                      className="form-input rounded-md border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white text-sm py-1 px-2 w-full" 
                      placeholder="Enter email address"
                      defaultValue="admin@example.com"
                    />
                  </div>
                </div>
                
                <div className="border rounded-lg p-4 dark:border-gray-700">
                  <div className="flex justify-between items-center mb-2">
                    <h4 className="font-medium text-gray-900 dark:text-white">Slack Notifications</h4>
                    <div className="relative inline-block w-12 h-6 border rounded-full dark:border-gray-700">
                      <input type="checkbox" className="sr-only" defaultChecked />
                      <span className="block absolute left-1 top-1 bg-indigo-600 dark:bg-indigo-500 w-4 h-4 rounded-full transition-transform transform translate-x-6"></span>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    Receive system alerts via Slack
                  </p>
                  <div className="flex items-center">
                    <input 
                      type="text" 
                      className="form-input rounded-md border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white text-sm py-1 px-2 w-full" 
                      placeholder="Webhook URL"
                      defaultValue="https://hooks.slack.com/services/XXX/YYY/ZZZ"
                    />
                  </div>
                </div>
                
                <div className="border rounded-lg p-4 dark:border-gray-700">
                  <div className="flex justify-between items-center mb-2">
                    <h4 className="font-medium text-gray-900 dark:text-white">Webhook</h4>
                    <div className="relative inline-block w-12 h-6 border rounded-full dark:border-gray-700">
                      <input type="checkbox" className="sr-only" />
                      <span className="block absolute left-1 top-1 bg-gray-300 dark:bg-gray-600 w-4 h-4 rounded-full transition-transform"></span>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    Send alerts to custom webhook
                  </p>
                  <div className="flex items-center">
                    <input 
                      type="text" 
                      className="form-input rounded-md border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white text-sm py-1 px-2 w-full" 
                      placeholder="Webhook URL"
                    />
                  </div>
                </div>
              </div>
            </div>
            
            <div className="flex space-x-3 pt-4 border-t dark:border-gray-700">
              <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 transition">
                Save Configuration
              </button>
              <button className="px-4 py-2 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600 transition">
                Test Alerts
              </button>
            </div>
          </Card>
        </>
      )}
    </div>
  );
};

export default LogsPanel;
