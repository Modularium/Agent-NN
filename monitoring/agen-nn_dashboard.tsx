import React, { useState } from 'react';
import { Home, Settings, BarChart, Database, Users, Code, Server, Shield, AlertTriangle, Book, RefreshCw, GitBranch, Cpu } from 'lucide-react';

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('system');

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-indigo-700 text-white p-4 shadow-md">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h1 className="text-2xl font-bold">Agent-NN Dashboard</h1>
            <span className="bg-indigo-600 text-sm px-2 py-1 rounded">v2.0.0</span>
          </div>
          <div className="flex items-center space-x-4">
            <button className="flex items-center space-x-2 bg-indigo-600 px-3 py-2 rounded hover:bg-indigo-500">
              <RefreshCw size={16} />
              <span>Refresh</span>
            </button>
            <div className="flex items-center space-x-2">
              <span>Admin</span>
              <div className="w-8 h-8 bg-indigo-500 rounded-full flex items-center justify-center">
                <span>A</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-64 bg-white shadow-md">
          <nav className="p-4">
            <ul className="space-y-2">
              <li>
                <button 
                  className={`flex items-center space-x-3 w-full p-3 rounded-md ${activeTab === 'system' ? 'bg-indigo-100 text-indigo-700' : 'hover:bg-gray-100'}`}
                  onClick={() => setActiveTab('system')}
                >
                  <Server size={20} />
                  <span>System Overview</span>
                </button>
              </li>
              <li>
                <button 
                  className={`flex items-center space-x-3 w-full p-3 rounded-md ${activeTab === 'agents' ? 'bg-indigo-100 text-indigo-700' : 'hover:bg-gray-100'}`}
                  onClick={() => setActiveTab('agents')}
                >
                  <Users size={20} />
                  <span>Agents</span>
                </button>
              </li>
              <li>
                <button 
                  className={`flex items-center space-x-3 w-full p-3 rounded-md ${activeTab === 'models' ? 'bg-indigo-100 text-indigo-700' : 'hover:bg-gray-100'}`}
                  onClick={() => setActiveTab('models')}
                >
                  <Cpu size={20} />
                  <span>Models</span>
                </button>
              </li>
              <li>
                <button 
                  className={`flex items-center space-x-3 w-full p-3 rounded-md ${activeTab === 'knowledge' ? 'bg-indigo-100 text-indigo-700' : 'hover:bg-gray-100'}`}
                  onClick={() => setActiveTab('knowledge')}
                >
                  <Database size={20} />
                  <span>Knowledge Bases</span>
                </button>
              </li>
              <li>
                <button 
                  className={`flex items-center space-x-3 w-full p-3 rounded-md ${activeTab === 'monitoring' ? 'bg-indigo-100 text-indigo-700' : 'hover:bg-gray-100'}`}
                  onClick={() => setActiveTab('monitoring')}
                >
                  <BarChart size={20} />
                  <span>Monitoring</span>
                </button>
              </li>
              <li>
                <button 
                  className={`flex items-center space-x-3 w-full p-3 rounded-md ${activeTab === 'security' ? 'bg-indigo-100 text-indigo-700' : 'hover:bg-gray-100'}`}
                  onClick={() => setActiveTab('security')}
                >
                  <Shield size={20} />
                  <span>Security</span>
                </button>
              </li>
              <li>
                <button 
                  className={`flex items-center space-x-3 w-full p-3 rounded-md ${activeTab === 'testing' ? 'bg-indigo-100 text-indigo-700' : 'hover:bg-gray-100'}`}
                  onClick={() => setActiveTab('testing')}
                >
                  <GitBranch size={20} />
                  <span>A/B Testing</span>
                </button>
              </li>
              <li>
                <button 
                  className={`flex items-center space-x-3 w-full p-3 rounded-md ${activeTab === 'settings' ? 'bg-indigo-100 text-indigo-700' : 'hover:bg-gray-100'}`}
                  onClick={() => setActiveTab('settings')}
                >
                  <Settings size={20} />
                  <span>Settings</span>
                </button>
              </li>
              <li>
                <button 
                  className={`flex items-center space-x-3 w-full p-3 rounded-md ${activeTab === 'logs' ? 'bg-indigo-100 text-indigo-700' : 'hover:bg-gray-100'}`}
                  onClick={() => setActiveTab('logs')}
                >
                  <AlertTriangle size={20} />
                  <span>Logs & Alerts</span>
                </button>
              </li>
              <li>
                <button 
                  className={`flex items-center space-x-3 w-full p-3 rounded-md ${activeTab === 'docs' ? 'bg-indigo-100 text-indigo-700' : 'hover:bg-gray-100'}`}
                  onClick={() => setActiveTab('docs')}
                >
                  <Book size={20} />
                  <span>Documentation</span>
                </button>
              </li>
            </ul>
          </nav>
        </aside>

        {/* Content Area */}
        <main className="flex-1 overflow-auto p-6">
          {activeTab === 'system' && <SystemOverview />}
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
      </div>
    </div>
  );
};

const SystemOverview = () => {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold">System Overview</h2>
        <div className="flex space-x-2">
          <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm flex items-center">
            <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
            System Online
          </span>
        </div>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-lg shadow">
          <div className="flex items-center justify-between">
            <h3 className="text-gray-500">CPU Usage</h3>
            <span className="text-green-500">Normal</span>
          </div>
          <p className="text-2xl font-bold mt-2">24%</p>
          <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
            <div className="bg-green-500 h-2.5 rounded-full" style={{ width: '24%' }}></div>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow">
          <div className="flex items-center justify-between">
            <h3 className="text-gray-500">Memory Usage</h3>
            <span className="text-yellow-500">Warning</span>
          </div>
          <p className="text-2xl font-bold mt-2">78%</p>
          <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
            <div className="bg-yellow-500 h-2.5 rounded-full" style={{ width: '78%' }}></div>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow">
          <div className="flex items-center justify-between">
            <h3 className="text-gray-500">GPU Usage</h3>
            <span className="text-green-500">Normal</span>
          </div>
          <p className="text-2xl font-bold mt-2">42%</p>
          <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
            <div className="bg-green-500 h-2.5 rounded-full" style={{ width: '42%' }}></div>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow">
          <div className="flex items-center justify-between">
            <h3 className="text-gray-500">Disk Space</h3>
            <span className="text-green-500">Normal</span>
          </div>
          <p className="text-2xl font-bold mt-2">36%</p>
          <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
            <div className="bg-green-500 h-2.5 rounded-full" style={{ width: '36%' }}></div>
          </div>
        </div>
      </div>

      {/* Active Tasks & Components */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="font-bold mb-4">Active Tasks</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
              <tr className="bg-gray-50">
                <th className="p-2 text-left">Name</th>
                <th className="p-2 text-left">Type</th>
                <th className="p-2 text-left">Source</th>
                <th className="p-2 text-left">Version</th>
                <th className="p-2 text-left">Status</th>
                <th className="p-2 text-left">Usage</th>
                <th className="p-2 text-left">Avg Latency</th>
                <th className="p-2 text-left">Actions</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-t">
                <td className="p-2 font-medium">gpt-4</td>
                <td className="p-2">LLM</td>
                <td className="p-2">OpenAI</td>
                <td className="p-2">v1.0</td>
                <td className="p-2"><span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Active</span></td>
                <td className="p-2">32,145 requests</td>
                <td className="p-2">1.2s</td>
                <td className="p-2">
                  <div className="flex space-x-2">
                    <button className="p-1 text-gray-500 hover:text-indigo-600">Configure</button>
                    <button className="p-1 text-gray-500 hover:text-red-600">Disable</button>
                  </div>
                </td>
              </tr>
              <tr className="border-t">
                <td className="p-2 font-medium">claude-3</td>
                <td className="p-2">LLM</td>
                <td className="p-2">Anthropic</td>
                <td className="p-2">v1.0</td>
                <td className="p-2"><span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Active</span></td>
                <td className="p-2">18,921 requests</td>
                <td className="p-2">1.5s</td>
                <td className="p-2">
                  <div className="flex space-x-2">
                    <button className="p-1 text-gray-500 hover:text-indigo-600">Configure</button>
                    <button className="p-1 text-gray-500 hover:text-red-600">Disable</button>
                  </div>
                </td>
              </tr>
              <tr className="border-t">
                <td className="p-2 font-medium">llama-3</td>
                <td className="p-2">LLM</td>
                <td className="p-2">Local</td>
                <td className="p-2">v1.0</td>
                <td className="p-2"><span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Active</span></td>
                <td className="p-2">8,752 requests</td>
                <td className="p-2">2.1s</td>
                <td className="p-2">
                  <div className="flex space-x-2">
                    <button className="p-1 text-gray-500 hover:text-indigo-600">Configure</button>
                    <button className="p-1 text-gray-500 hover:text-red-600">Disable</button>
                  </div>
                </td>
              </tr>
            </tbody>
                <tr className="bg-gray-50">
                  <th className="p-2 text-left">ID</th>
                  <th className="p-2 text-left">Type</th>
                  <th className="p-2 text-left">Agent</th>
                  <th className="p-2 text-left">Status</th>
                  <th className="p-2 text-left">Duration</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-t">
                  <td className="p-2">T-1254</td>
                  <td className="p-2">Analysis</td>
                  <td className="p-2">Finance</td>
                  <td className="p-2"><span className="px-2 py-1 bg-yellow-100 text-yellow-800 rounded text-xs">Running</span></td>
                  <td className="p-2">24s</td>
                </tr>
                <tr className="border-t">
                  <td className="p-2">T-1253</td>
                  <td className="p-2">Research</td>
                  <td className="p-2">Web</td>
                  <td className="p-2"><span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Completed</span></td>
                  <td className="p-2">3m 12s</td>
                </tr>
                <tr className="border-t">
                  <td className="p-2">T-1252</td>
                  <td className="p-2">Code</td>
                  <td className="p-2">Tech</td>
                  <td className="p-2"><span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">Queued</span></td>
                  <td className="p-2">-</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="font-bold mb-4">System Components</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="bg-gray-50">
                  <th className="p-2 text-left">Component</th>
                  <th className="p-2 text-left">Status</th>
                  <th className="p-2 text-left">Version</th>
                  <th className="p-2 text-left">Last Updated</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-t">
                  <td className="p-2">Supervisor Agent</td>
                  <td className="p-2"><span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Online</span></td>
                  <td className="p-2">2.1.0</td>
                  <td className="p-2">2 hours ago</td>
                </tr>
                <tr className="border-t">
                  <td className="p-2">MLflow Integration</td>
                  <td className="p-2"><span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Online</span></td>
                  <td className="p-2">1.4.2</td>
                  <td className="p-2">1 day ago</td>
                </tr>
                <tr className="border-t">
                  <td className="p-2">Vector Store</td>
                  <td className="p-2"><span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Online</span></td>
                  <td className="p-2">3.0.1</td>
                  <td className="p-2">3 days ago</td>
                </tr>
                <tr className="border-t">
                  <td className="p-2">Cache Manager</td>
                  <td className="p-2"><span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Online</span></td>
                  <td className="p-2">1.2.5</td>
                  <td className="p-2">5 days ago</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* System Configuration */}
      <div className="bg-white p-4 rounded-lg shadow">
        <h3 className="font-bold mb-4">System Configuration</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">Maximum Concurrent Tasks</label>
              <div className="flex items-center">
                <input type="number" className="form-input rounded-md w-24 border-gray-300" value="10" />
                <button className="ml-2 px-3 py-1 bg-indigo-100 text-indigo-700 rounded text-sm">Update</button>
              </div>
            </div>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">Task Timeout (seconds)</label>
              <div className="flex items-center">
                <input type="number" className="form-input rounded-md w-24 border-gray-300" value="300" />
                <button className="ml-2 px-3 py-1 bg-indigo-100 text-indigo-700 rounded text-sm">Update</button>
              </div>
            </div>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">Logging Level</label>
              <div className="flex items-center">
                <select className="form-select rounded-md w-36 border-gray-300">
                  <option>INFO</option>
                  <option>DEBUG</option>
                  <option>WARNING</option>
                  <option>ERROR</option>
                </select>
                <button className="ml-2 px-3 py-1 bg-indigo-100 text-indigo-700 rounded text-sm">Update</button>
              </div>
            </div>
          </div>
          <div>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">Cache Size (MB)</label>
              <div className="flex items-center">
                <input type="number" className="form-input rounded-md w-24 border-gray-300" value="1024" />
                <button className="ml-2 px-3 py-1 bg-indigo-100 text-indigo-700 rounded text-sm">Update</button>
              </div>
            </div>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">Monitoring Interval (seconds)</label>
              <div className="flex items-center">
                <input type="number" className="form-input rounded-md w-24 border-gray-300" value="60" />
                <button className="ml-2 px-3 py-1 bg-indigo-100 text-indigo-700 rounded text-sm">Update</button>
              </div>
            </div>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">Backup Interval (seconds)</label>
              <div className="flex items-center">
                <input type="number" className="form-input rounded-md w-24 border-gray-300" value="86400" />
                <button className="ml-2 px-3 py-1 bg-indigo-100 text-indigo-700 rounded text-sm">Update</button>
              </div>
            </div>
          </div>
        </div>

        <div className="flex space-x-3 mt-4">
          <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700">Save All Changes</button>
          <button className="px-4 py-2 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300">Reset to Defaults</button>
          <button className="px-4 py-2 bg-red-100 text-red-700 rounded-md hover:bg-red-200">Clear Cache</button>
        </div>
      </div>
    </div>
  );
};

const AgentsPanel = () => {
  const [selectedTab, setSelectedTab] = useState('active');
  
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold">Agent Management</h2>
        <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 flex items-center space-x-2">
          <span>Create New Agent</span>
        </button>
      </div>

      <div className="bg-white rounded-lg shadow">
        <nav className="flex border-b">
          <button 
            className={`px-6 py-3 font-medium ${selectedTab === 'active' ? 'border-b-2 border-indigo-600 text-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
            onClick={() => setSelectedTab('active')}
          >
            Active Agents
          </button>
          <button 
            className={`px-6 py-3 font-medium ${selectedTab === 'custom' ? 'border-b-2 border-indigo-600 text-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
            onClick={() => setSelectedTab('custom')}
          >
            Custom Agents
          </button>
          <button 
            className={`px-6 py-3 font-medium ${selectedTab === 'templates' ? 'border-b-2 border-indigo-600 text-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
            onClick={() => setSelectedTab('templates')}
          >
            Agent Templates
          </button>
          <button 
            className={`px-6 py-3 font-medium ${selectedTab === 'stats' ? 'border-b-2 border-indigo-600 text-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
            onClick={() => setSelectedTab('stats')}
          >
            Performance Stats
          </button>
        </nav>

        {selectedTab === 'active' && (
          <div className="p-4">
            <div className="overflow-x-auto">
              <table className="min-w-full">
                <thead>
                  <tr className="bg-gray-50">
                    <th className="p-2 text-left">Name</th>
                    <th className="p-2 text-left">Domain</th>
                    <th className="p-2 text-left">Status</th>
                    <th className="p-2 text-left">Tasks</th>
                    <th className="p-2 text-left">Success Rate</th>
                    <th className="p-2 text-left">Avg Response</th>
                    <th className="p-2 text-left">Last Active</th>
                    <th className="p-2 text-left">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-t">
                    <td className="p-2 font-medium">Finance Agent</td>
                    <td className="p-2">Finance</td>
                    <td className="p-2"><span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Active</span></td>
                    <td className="p-2">1,245</td>
                    <td className="p-2">92.5%</td>
                    <td className="p-2">1.2s</td>
                    <td className="p-2">Just now</td>
                    <td className="p-2">
                      <div className="flex space-x-2">
                        <button className="p-1 text-gray-500 hover:text-indigo-600">Edit</button>
                        <button className="p-1 text-gray-500 hover:text-red-600">Disable</button>
                      </div>
                    </td>
                  </tr>
                  <tr className="border-t">
                    <td className="p-2 font-medium">Tech Agent</td>
                    <td className="p-2">Technology</td>
                    <td className="p-2"><span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Active</span></td>
                    <td className="p-2">2,153</td>
                    <td className="p-2">94.1%</td>
                    <td className="p-2">0.9s</td>
                    <td className="p-2">2 minutes ago</td>
                    <td className="p-2">
                      <div className="flex space-x-2">
                        <button className="p-1 text-gray-500 hover:text-indigo-600">Edit</button>
                        <button className="p-1 text-gray-500 hover:text-red-600">Disable</button>
                      </div>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {selectedTab === 'custom' && (
          <div className="p-4">
            <div className="overflow-x-auto">
              <table className="min-w-full">
                <thead>
                  <tr className="bg-gray-50">
                    <th className="p-2 text-left">Name</th>
                    <th className="p-2 text-left">Domain</th>
                    <th className="p-2 text-left">Created</th>
                    <th className="p-2 text-left">Status</th>
                    <th className="p-2 text-left">Base Model</th>
                    <th className="p-2 text-left">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-t">
                    <td className="p-2 font-medium">Healthcare Agent</td>
                    <td className="p-2">Healthcare</td>
                    <td className="p-2">3 days ago</td>
                    <td className="p-2"><span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Active</span></td>
                    <td className="p-2">gpt-4</td>
                    <td className="p-2">
                      <div className="flex space-x-2">
                        <button className="p-1 text-gray-500 hover:text-indigo-600">Edit</button>
                        <button className="p-1 text-gray-500 hover:text-red-600">Delete</button>
                      </div>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {selectedTab === 'templates' && (
          <div className="p-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="border rounded-lg p-4">
                <h3 className="font-bold">Finance Specialist</h3>
                <p className="text-sm text-gray-600 mt-2">Specialized in financial analysis, investment strategies, and market trends.</p>
                <div className="mt-4 space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Base Model</span>
                    <span>gpt-4</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Knowledge Base</span>
                    <span>Finance</span>
                  </div>
                </div>
                <button className="mt-4 w-full py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700">Use Template</button>
              </div>

              <div className="border rounded-lg p-4">
                <h3 className="font-bold">Tech Specialist</h3>
                <p className="text-sm text-gray-600 mt-2">Specialized in programming, system architecture, and technical troubleshooting.</p>
                <div className="mt-4 space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Base Model</span>
                    <span>claude-3</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Knowledge Base</span>
                    <span>Technology</span>
                  </div>
                </div>
                <button className="mt-4 w-full py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700">Use Template</button>
              </div>
            </div>
          </div>
        )}

        {selectedTab === 'stats' && (
          <div className="p-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="border rounded-lg p-4">
                <h3 className="font-bold">Agent Success Rate</h3>
                <div className="mt-4 space-y-4">
                  <div>
                    <div className="flex justify-between mb-1">
                      <span>Finance Agent</span>
                      <span>92.5%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                      <div className="bg-green-500 h-2.5 rounded-full" style={{ width: '92.5%' }}></div>
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between mb-1">
                      <span>Tech Agent</span>
                      <span>94.1%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                      <div className="bg-green-500 h-2.5 rounded-full" style={{ width: '94.1%' }}></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const ModelsPanel = () => {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold">Model Management</h2>
        <div className="flex space-x-3">
          <button className="px-4 py-2 bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200">Import Model</button>
          <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700">Add New Model</button>
        </div>
      </div>

      {/* Model List */}
      <div className="bg-white p-4 rounded-lg shadow overflow-hidden">
        <h3 className="font-bold mb-4">Available Models</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full">
            <thead>
