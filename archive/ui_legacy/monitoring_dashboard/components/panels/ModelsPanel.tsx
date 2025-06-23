// monitoring/dashboard/components/panels/ModelsPanel.tsx
import React, { useState } from 'react';
import { useDashboard } from '../../context/DashboardContext';
import Card from '../common/Card';
import StatusBadge from '../common/StatusBadge';
import { Database, Cloud, Upload } from 'lucide-react';

const ModelsPanel: React.FC = () => {
  const { models } = useDashboard();
  const [activeTab, setActiveTab] = useState<'list' | 'performance' | 'settings'>('list');

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">Model Management</h2>
        <div className="flex space-x-3">
          <button className="flex items-center space-x-2 px-4 py-2 bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200 dark:bg-blue-900/30 dark:text-blue-300 dark:hover:bg-blue-900/50 transition">
            <Upload size={16} />
            <span>Import Model</span>
          </button>
          <button className="flex items-center space-x-2 px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 transition">
            <Database size={16} />
            <span>Add New Model</span>
          </button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-gray-200 dark:border-gray-700">
        <button
          className={`px-4 py-2 font-medium ${
            activeTab === 'list'
              ? 'border-b-2 border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('list')}
        >
          Available Models
        </button>
        <button
          className={`px-4 py-2 font-medium ${
            activeTab === 'performance'
              ? 'border-b-2 border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('performance')}
        >
          Performance
        </button>
        <button
          className={`px-4 py-2 font-medium ${
            activeTab === 'settings'
              ? 'border-b-2 border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('settings')}
        >
          Settings
        </button>
      </div>

      {/* Model List Tab */}
      {activeTab === 'list' && (
        <Card>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead>
                <tr>
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Name</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Type</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Source</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Version</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Usage</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Avg Latency</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {models.map((model, index) => (
                  <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                    <td className="px-3 py-2 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">{model.name}</td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{model.type}</td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      <div className="flex items-center">
                        {model.source === 'OpenAI' && <Cloud size={16} className="mr-1 text-green-500" />}
                        {model.source === 'Anthropic' && <Cloud size={16} className="mr-1 text-blue-500" />}
                        {model.source === 'Local' && <Database size={16} className="mr-1 text-purple-500" />}
                        {model.source}
                      </div>
                    </td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{model.version}</td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm">
                      <StatusBadge status={model.status} />
                    </td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{model.requests.toLocaleString()} requests</td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{model.latency}s</td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm">
                      <div className="flex space-x-2">
                        <button className="text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-300">Configure</button>
                        <button className="text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300">Disable</button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Performance Tab */}
      {activeTab === 'performance' && (
        <div className="grid grid-cols-1 gap-6">
          <Card title="Model Performance Comparison">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="border rounded-lg p-4 dark:border-gray-700">
                <h4 className="font-bold text-gray-900 dark:text-white mb-4">Response Time Comparison</h4>
                <div className="h-64 flex items-end space-x-4 pb-6 px-2">
                  {models.map((model, index) => (
                    <div key={index} className="flex flex-col items-center flex-grow">
                      <div 
                        className={`w-full rounded-t-md ${
                          model.source === 'OpenAI' ? 'bg-green-500' : 
                          model.source === 'Anthropic' ? 'bg-blue-500' : 'bg-purple-500'
                        }`} 
                        style={{ height: `${Math.max(5, 100 - (model.latency * 30))}%` }}
                      ></div>
                      <div className="text-xs mt-2 truncate w-full text-center text-gray-700 dark:text-gray-300">{model.name}</div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">{model.latency}s</div>
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="border rounded-lg p-4 dark:border-gray-700">
                <h4 className="font-bold text-gray-900 dark:text-white mb-4">Model Usage</h4>
                <div className="h-64 flex flex-col justify-center">
                  <div className="relative h-48 w-48 mx-auto">
                    {/* Pie chart segments (simplified representation) */}
                    <svg viewBox="0 0 100 100" className="h-full w-full">
                      {models.map((model, index) => {
                        // Calculate percentage of total requests
                        const totalRequests = models.reduce((sum, m) => sum + m.requests, 0);
                        const percentage = (model.requests / totalRequests) * 100;
                        
                        // Simple approach to create pie segments (not mathematically precise but visual representation)
                        const colors = ['#10B981', '#3B82F6', '#8B5CF6'];
                        const offset = index === 0 ? 0 : models.slice(0, index).reduce((sum, m) => sum + (m.requests / totalRequests) * 100, 0);
                        
                        return (
                          <circle 
                            key={index}
                            cx="50" 
                            cy="50" 
                            r="40"
                            stroke={colors[index % colors.length]}
                            strokeWidth="20"
                            fill="none"
                            strokeDasharray={`${percentage} ${100 - percentage}`}
                            strokeDashoffset={`-${offset}`}
                            transform="rotate(-90 50 50)"
                          />
                        );
                      })}
                    </svg>
                    
                    {/* Total count in center */}
                    <div className="absolute inset-0 flex items-center justify-center font-bold text-lg text-gray-900 dark:text-white">
                      {models.reduce((total, model) => total + model.requests, 0).toLocaleString()}
                    </div>
                  </div>
                  
                  {/* Legend */}
                  <div className="flex flex-col space-y-2 mt-4">
                    {models.map((model, index) => {
                      const colors = ['#10B981', '#3B82F6', '#8B5CF6'];
                      return (
                        <div key={index} className="flex items-center">
                          <div className="w-4 h-4 rounded-full mr-2" style={{ backgroundColor: colors[index % colors.length] }}></div>
                          <span className="text-sm text-gray-700 dark:text-gray-300">{model.name}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          </Card>
          
          <Card title="Historical Performance">
            <div className="h-64 border border-gray-200 dark:border-gray-700 rounded-lg p-4 flex items-center justify-center">
              <p className="text-gray-500 dark:text-gray-400">Line chart showing historical performance metrics over time would be displayed here</p>
            </div>
          </Card>
        </div>
      )}

      {/* Settings Tab */}
      {activeTab === 'settings' && (
        <Card title="Global Model Settings">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Default Model</label>
                <div className="flex items-center">
                  <select className="form-select rounded-md w-48 border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white">
                    {models.map((model, index) => (
                      <option key={index}>{model.name}</option>
                    ))}
                  </select>
                  <button className="ml-2 px-3 py-1 bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-400 rounded text-sm">
                    Update
                  </button>
                </div>
              </div>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">API Key Management</label>
                <div className="flex items-center">
                  <button className="px-3 py-1 bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-400 rounded text-sm">
                    Manage API Keys
                  </button>
                </div>
              </div>
            </div>
            <div>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Model Cache TTL (seconds)</label>
                <div className="flex items-center">
                  <input type="number" className="form-input rounded-md w-24 border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white" defaultValue="3600" />
                  <button className="ml-2 px-3 py-1 bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-400 rounded text-sm">
                    Update
                  </button>
                </div>
              </div>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Maximum Tokens</label>
                <div className="flex items-center">
                  <input type="number" className="form-input rounded-md w-24 border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white" defaultValue="4096" />
                  <button className="ml-2 px-3 py-1 bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-400 rounded text-sm">
                    Update
                  </button>
                </div>
              </div>
            </div>
          </div>
          
          <div className="pt-4 border-t border-gray-200 dark:border-gray-700 mt-4">
            <h3 className="font-bold text-gray-900 dark:text-white mb-4">Model Advanced Settings</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Temperature</label>
                  <div className="flex items-center">
                    <input 
                      type="range" 
                      min="0" 
                      max="1" 
                      step="0.1" 
                      defaultValue="0.7" 
                      className="w-48 mr-2"
                    />
                    <span className="text-gray-700 dark:text-gray-300">0.7</span>
                  </div>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Controls randomness: Lower values are more deterministic, higher values more creative</p>
                </div>
                
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Top P</label>
                  <div className="flex items-center">
                    <input 
                      type="range" 
                      min="0" 
                      max="1" 
                      step="0.1" 
                      defaultValue="0.9" 
                      className="w-48 mr-2"
                    />
                    <span className="text-gray-700 dark:text-gray-300">0.9</span>
                  </div>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Alternative to temperature, considers tokens with top_p probability mass</p>
                </div>
              </div>
              
              <div>
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Frequency Penalty</label>
                  <div className="flex items-center">
                    <input 
                      type="range" 
                      min="0" 
                      max="2" 
                      step="0.1" 
                      defaultValue="0.0" 
                      className="w-48 mr-2"
                    />
                    <span className="text-gray-700 dark:text-gray-300">0.0</span>
                  </div>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Reduces repetition by penalizing tokens based on frequency</p>
                </div>
                
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Presence Penalty</label>
                  <div className="flex items-center">
                    <input 
                      type="range" 
                      min="0" 
                      max="2" 
                      step="0.1" 
                      defaultValue="0.0" 
                      className="w-48 mr-2"
                    />
                    <span className="text-gray-700 dark:text-gray-300">0.0</span>
                  </div>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Reduces repetition by penalizing tokens that have appeared at all</p>
                </div>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Add New Model Form */}
      <Card title="Add New Model">
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Model Name</label>
              <input 
                type="text" 
                className="form-input rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
                placeholder="e.g. gpt-4-turbo"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Model Type</label>
              <select className="form-select rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white">
                <option>LLM</option>
                <option>Embedding</option>
                <option>Vision</option>
                <option>Speech</option>
                <option>Multimodal</option>
                <option>NN</option>
                <option>Other</option>
              </select>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Source</label>
              <select className="form-select rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white">
                <option>OpenAI</option>
                <option>Anthropic</option>
                <option>Local</option>
                <option>Hugging Face</option>
                <option>Azure</option>
                <option>Other</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Version</label>
              <input 
                type="text" 
                className="form-input rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
                placeholder="e.g. v1.0"
              />
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">API Key (if applicable)</label>
              <input 
                type="password" 
                className="form-input rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
                placeholder="Enter API key"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Model Endpoint (if applicable)</label>
              <input 
                type="text" 
                className="form-input rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
                placeholder="e.g. https://api.example.com/v1"
              />
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Model Path (for local models)</label>
            <div className="flex items-center">
              <input 
                type="text" 
                className="form-input rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
                placeholder="e.g. /path/to/model"
              />
              <button className="ml-2 px-3 py-1 bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-400 rounded text-sm">
                Browse
              </button>
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Model Description</label>
            <textarea 
              className="form-textarea rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
              rows={3}
              placeholder="Add a description for this model..."
            ></textarea>
          </div>
          
          <div className="flex space-x-3 pt-4">
            <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 transition">
              Add Model
            </button>
            <button className="px-4 py-2 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600 transition">
              Cancel
            </button>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default ModelsPanel;
