// monitoring/dashboard/components/panels/AgentsPanel.tsx
import React, { useState } from 'react';
import { Plus } from 'lucide-react';
import { useDashboard } from '../../context/DashboardContext';
import Card from '../common/Card';
import StatusBadge from '../common/StatusBadge';
import LoadingSpinner from '../common/LoadingSpinner';

const AgentsPanel: React.FC = () => {
  const { agents } = useDashboard();
  const [selectedTab, setSelectedTab] = useState('active');
  const [loading, setLoading] = useState(false);

  // Mock agent templates data
  const agentTemplates = [
    {
      id: 'tmpl-1',
      name: 'Finance Specialist',
      domain: 'Finance',
      description: 'Specialized in financial analysis, investment strategies, and market trends.',
      baseModel: 'gpt-4',
      knowledgeBase: 'Finance'
    },
    {
      id: 'tmpl-2',
      name: 'Tech Specialist',
      domain: 'Technology',
      description: 'Specialized in programming, system architecture, and technical troubleshooting.',
      baseModel: 'claude-3',
      knowledgeBase: 'Technology'
    },
    {
      id: 'tmpl-3',
      name: 'Marketing Specialist',
      domain: 'Marketing',
      description: 'Specialized in digital marketing, brand development, and market research.',
      baseModel: 'llama-3',
      knowledgeBase: 'Marketing'
    }
  ];

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">Agent Management</h2>
        <button className="flex items-center space-x-2 bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 transition">
          <Plus size={16} />
          <span>Create New Agent</span>
        </button>
      </div>

      <Card>
        <nav className="flex border-b border-gray-200 dark:border-gray-700">
          <button 
            className={`px-6 py-3 font-medium ${
              selectedTab === 'active' 
                ? 'border-b-2 border-indigo-600 text-indigo-600 dark:text-indigo-400 dark:border-indigo-400' 
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
            onClick={() => setSelectedTab('active')}
          >
            Active Agents
          </button>
          <button 
            className={`px-6 py-3 font-medium ${
              selectedTab === 'custom' 
                ? 'border-b-2 border-indigo-600 text-indigo-600 dark:text-indigo-400 dark:border-indigo-400' 
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
            onClick={() => setSelectedTab('custom')}
          >
            Custom Agents
          </button>
          <button 
            className={`px-6 py-3 font-medium ${
              selectedTab === 'templates' 
                ? 'border-b-2 border-indigo-600 text-indigo-600 dark:text-indigo-400 dark:border-indigo-400' 
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
            onClick={() => setSelectedTab('templates')}
          >
            Agent Templates
          </button>
          <button 
            className={`px-6 py-3 font-medium ${
              selectedTab === 'stats' 
                ? 'border-b-2 border-indigo-600 text-indigo-600 dark:text-indigo-400 dark:border-indigo-400' 
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
            onClick={() => setSelectedTab('stats')}
          >
            Performance Stats
          </button>
        </nav>

        {loading ? (
          <div className="py-10">
            <LoadingSpinner text="Loading agent data..." />
          </div>
        ) : (
          <>
            {selectedTab === 'active' && (
              <div className="p-4">
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead>
                      <tr>
                        <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Name</th>
                        <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Domain</th>
                        <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</th>
                        <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Tasks</th>
                        <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Success Rate</th>
                        <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Avg Response</th>
                        <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Last Active</th>
                        <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                      {agents.map((agent, index) => (
                        <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                          <td className="px-3 py-2 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">{agent.name}</td>
                          <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{agent.domain}</td>
                          <td className="px-3 py-2 whitespace-nowrap text-sm">
                            <StatusBadge status={agent.status} />
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{agent.tasks.toLocaleString()}</td>
                          <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{agent.successRate}%</td>
                          <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{agent.avgResponse}s</td>
                          <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{agent.lastActive}</td>
                          <td className="px-3 py-2 whitespace-nowrap text-sm">
                            <div className="flex space-x-2">
                              <button className="text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-300">Edit</button>
                              <button className="text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300">Disable</button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {selectedTab === 'custom' && (
              <div className="p-4">
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead>
                      <tr>
                        <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Name</th>
                        <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Domain</th>
                        <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Created</th>
                        <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</th>
                        <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Base Model</th>
                        <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                      <tr className="hover:bg-gray-50 dark:hover:bg-gray-800">
                        <td className="px-3 py-2 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">Healthcare Agent</td>
                        <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">Healthcare</td>
                        <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">3 days ago</td>
                        <td className="px-3 py-2 whitespace-nowrap text-sm">
                          <StatusBadge status="Active" />
                        </td>
                        <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">gpt-4</td>
                        <td className="px-3 py-2 whitespace-nowrap text-sm">
                          <div className="flex space-x-2">
                            <button className="text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-300">Edit</button>
                            <button className="text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300">Delete</button>
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
                  {agentTemplates.map(template => (
                    <div key={template.id} className="border rounded-lg p-4 hover:border-indigo-500 dark:border-gray-700 dark:hover:border-indigo-500 transition duration-150">
                      <h3 className="font-bold text-gray-900 dark:text-white">{template.name}</h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">{template.description}</p>
                      <div className="mt-4 space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-500 dark:text-gray-400">Base Model</span>
                          <span className="text-gray-700 dark:text-gray-300">{template.baseModel}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-500 dark:text-gray-400">Knowledge Base</span>
                          <span className="text-gray-700 dark:text-gray-300">{template.knowledgeBase}</span>
                        </div>
                      </div>
                      <button className="mt-4 w-full py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 transition">
                        Use Template
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {selectedTab === 'stats' && (
              <div className="p-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="border rounded-lg p-4 dark:border-gray-700">
                    <h3 className="font-bold text-gray-900 dark:text-white mb-4">Agent Success Rate</h3>
                    <div className="space-y-4">
                      {agents.map((agent, index) => (
                        <div key={index}>
                          <div className="flex justify-between mb-1">
                            <span className="text-gray-700 dark:text-gray-300">{agent.name}</span>
                            <span className="text-gray-700 dark:text-gray-300">{agent.successRate}%</span>
                          </div>
                          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                            <div 
                              className={`h-2.5 rounded-full ${
                                agent.successRate > 90 ? 'bg-green-500' : 
                                agent.successRate > 80 ? 'bg-yellow-500' : 'bg-red-500'
                              }`} 
                              style={{ width: `${agent.successRate}%` }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <div className="border rounded-lg p-4 dark:border-gray-700">
                    <h3 className="font-bold text-gray-900 dark:text-white mb-4">Average Response Time</h3>
                    <div className="space-y-4">
                      {agents.map((agent, index) => (
                        <div key={index}>
                          <div className="flex justify-between mb-1">
                            <span className="text-gray-700 dark:text-gray-300">{agent.name}</span>
                            <span className="text-gray-700 dark:text-gray-300">{agent.avgResponse}s</span>
                          </div>
                          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                            <div 
                              className={`h-2.5 rounded-full ${
                                agent.avgResponse < 1.0 ? 'bg-green-500' : 
                                agent.avgResponse < 2.0 ? 'bg-yellow-500' : 'bg-red-500'
                              }`} 
                              style={{ width: `${Math.min(100, agent.avgResponse * 33)}%` }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </Card>

      {/* Agent Creation Form */}
      <Card title="Create New Agent">
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Agent Name</label>
              <input 
                type="text" 
                className="form-input rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
                placeholder="e.g. Finance Expert"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Domain</label>
              <select className="form-select rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white">
                <option>Finance</option>
                <option>Technology</option>
                <option>Marketing</option>
                <option>Healthcare</option>
                <option>Research</option>
                <option>Legal</option>
                <option>Education</option>
                <option>Other</option>
              </select>
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Agent Description</label>
            <textarea 
              className="form-textarea rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
              rows={3}
              placeholder="Describe the agent's purpose and capabilities..."
            ></textarea>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Base Model</label>
              <select className="form-select rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white">
                <option>gpt-4</option>
                <option>claude-3</option>
                <option>llama-3</option>
                <option>mistral-7b</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Knowledge Base</label>
              <select className="form-select rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white">
                <option>Finance KB</option>
                <option>Tech KB</option>
                <option>Marketing KB</option>
                <option>General KB</option>
                <option>No Knowledge Base</option>
              </select>
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Capabilities</label>
            <div className="space-y-2">
              <div className="flex items-center">
                <input type="checkbox" className="mr-2" checked />
                <span className="text-gray-700 dark:text-gray-300">Data Analysis</span>
              </div>
              <div className="flex items-center">
                <input type="checkbox" className="mr-2" checked />
                <span className="text-gray-700 dark:text-gray-300">Query Generation</span>
              </div>
              <div className="flex items-center">
                <input type="checkbox" className="mr-2" checked />
                <span className="text-gray-700 dark:text-gray-300">Document Processing</span>
              </div>
              <div className="flex items-center">
                <input type="checkbox" className="mr-2" />
                <span className="text-gray-700 dark:text-gray-300">Web Search</span>
              </div>
              <div className="flex items-center">
                <input type="checkbox" className="mr-2" />
                <span className="text-gray-700 dark:text-gray-300">API Integration</span>
              </div>
            </div>
          </div>
          
          <div className="flex space-x-3 pt-4">
            <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 transition">
              Create Agent
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

export default AgentsPanel;
