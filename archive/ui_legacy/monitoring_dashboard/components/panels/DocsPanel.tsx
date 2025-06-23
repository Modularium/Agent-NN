// monitoring/dashboard/components/panels/DocsPanel.tsx
import React, { useState } from 'react';
import { Book, Code, Users, Search, FileText, ExternalLink } from 'lucide-react';
import Card from '../common/Card';

const DocsPanel: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [activeSection, setActiveSection] = useState<string | null>(null);
  
  const guideCategories = [
    { id: 'getting-started', name: 'Getting Started', icon: <Book size={20} className="text-indigo-600 dark:text-indigo-400 mr-2" />, count: 4 },
    { id: 'api-reference', name: 'API Reference', icon: <Code size={20} className="text-indigo-600 dark:text-indigo-400 mr-2" />, count: 12 },
    { id: 'agent-guides', name: 'Agent Creation Guide', icon: <Users size={20} className="text-indigo-600 dark:text-indigo-400 mr-2" />, count: 7 }
  ];
  
  const commonCommands = [
    { command: 'smolit login', description: 'Authenticate with the system' },
    { command: 'smolit task "Task description"', description: 'Submit a task for execution' },
    { command: 'smolit agents', description: 'List available agents' },
    { command: 'smolit metrics', description: 'Display system metrics' },
    { command: 'smolit kb list', description: 'List knowledge bases' },
    { command: 'smolit model list', description: 'List available models' }
  ];
  
  const apiExamples = [
    { 
      name: 'Submit a task via API',
      method: 'POST',
      endpoint: '/api/v2/tasks',
      body: `{
  "description": "Analyze market trends",
  "domain": "finance",
  "priority": 2
}`,
      response: `{
  "task_id": "t-12345",
  "status": "queued",
  "estimated_completion": "2025-03-16T15:30:00Z"
}`
    },
    { 
      name: 'Get task status',
      method: 'GET',
      endpoint: '/api/v2/tasks/{task_id}',
      response: `{
  "task_id": "t-12345",
  "status": "completed",
  "result": "Market analysis suggests growth in tech sector..."
}`
    }
  ];
  
  const documentationContents = [
    {
      section: 'System Overview',
      topics: ['Architecture', 'Components', 'Security', 'Monitoring']
    },
    {
      section: 'User Guides',
      topics: ['Installation Guide', 'Configuration', 'Creating Agents', 'Knowledge Management']
    },
    {
      section: 'API Reference',
      topics: ['Authentication', 'Task Management', 'Agent Management', 'System Management']
    },
    {
      section: 'Tutorials',
      topics: ['Building Custom Agents', 'Setting Up A/B Tests', 'Integrating External APIs', 'Performance Optimization']
    }
  ];
  
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">Documentation</h2>
        <div className="flex space-x-3">
          <div className="relative">
            <input 
              type="text" 
              className="pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white rounded-md focus:ring-indigo-500 focus:border-indigo-500" 
              placeholder="Search documentation..." 
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <div className="absolute left-3 top-2.5 text-gray-400">
              <Search size={18} />
            </div>
          </div>
          <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 transition">
            Search
          </button>
        </div>
      </div>

      {/* Documentation Overview */}
      <Card>
        <h3 className="font-bold text-lg mb-4 text-gray-900 dark:text-white">Welcome to Agent-NN Documentation</h3>
        <p className="text-gray-600 dark:text-gray-400 mb-4">
          This documentation provides comprehensive information about the Agent-NN system, including
          setup instructions, API references, and best practices for creating and managing agents.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          {guideCategories.map(category => (
            <div 
              key={category.id}
              className="border rounded-lg p-4 hover:border-indigo-500 dark:border-gray-700 dark:hover:border-indigo-500 cursor-pointer transition-colors"
              onClick={() => setActiveSection(category.id)}
            >
              <div className="flex items-center mb-2">
                {category.icon}
                <h4 className="font-semibold text-gray-900 dark:text-white">{category.name}</h4>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {category.id === 'getting-started' && 'Learn the basics of Agent-NN and how to set up your first agents.'}
                {category.id === 'api-reference' && 'Explore the Agent-NN API for programmatic interaction with the system.'}
                {category.id === 'agent-guides' && 'Best practices for designing and implementing effective agents.'}
              </p>
              <div className="flex justify-between items-center mt-2">
                <span className="text-indigo-600 dark:text-indigo-400 text-sm">
                  {category.count} articles
                </span>
                <span className="text-indigo-600 dark:text-indigo-400 text-sm">Read more →</span>
              </div>
            </div>
          ))}
        </div>
      </Card>
      
      {/* Quick Reference */}
      <Card title="Quick Reference">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-semibold mb-2 text-gray-900 dark:text-white">Common Commands</h4>
            <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-lg font-mono text-sm">
              {commonCommands.map((cmd, index) => (
                <div key={index} className="mb-2">
                  <span className="text-indigo-600 dark:text-indigo-400">smolit</span> <span className="text-red-500 dark:text-red-400">{cmd.command.split(' ')[1]}</span> {cmd.command.split(' ').slice(2).join(' ') && <span className="text-green-600 dark:text-green-400">{cmd.command.split(' ').slice(2).join(' ')}</span>}
                  <p className="text-gray-500 dark:text-gray-400 text-xs mt-1">{cmd.description}</p>
                </div>
              ))}
            </div>
          </div>
          
          <div>
            <h4 className="font-semibold mb-2 text-gray-900 dark:text-white">API Examples</h4>
            <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-lg font-mono text-sm">
              {apiExamples.map((example, index) => (
                <div key={index} className="mb-4">
                  <p className="text-gray-500 dark:text-gray-400">// {example.name}</p>
                  <span className="text-blue-600 dark:text-blue-400">{example.method}</span> <span>{example.endpoint}</span>
                  {example.body && (
                    <pre className="text-xs mt-1 text-green-700 dark:text-green-400 overflow-x-auto">{example.body}</pre>
                  )}
                  {example.response && (
                    <>
                      <p className="text-gray-500 dark:text-gray-400 mt-1">// Response:</p>
                      <pre className="text-xs mt-1 text-yellow-700 dark:text-yellow-400 overflow-x-auto">{example.response}</pre>
                    </>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </Card>
      
      {/* Documentation Table of Contents */}
      <Card title="Documentation Contents">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4">
          {documentationContents.map((section, index) => (
            <div key={index}>
              <h4 className="font-semibold border-b pb-2 mb-2 text-gray-900 dark:text-white dark:border-gray-700">{section.section}</h4>
              <ul className="space-y-1">
                {section.topics.map((topic, topicIndex) => (
                  <li key={topicIndex} className="text-indigo-600 dark:text-indigo-400 hover:underline cursor-pointer flex items-center">
                    <FileText size={14} className="mr-1" />
                    {topic}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </Card>
      
      {/* Recent Updates */}
      <Card title="Recent Documentation Updates">
        <ul className="space-y-3">
          <li className="border-b pb-3 dark:border-gray-700">
            <div className="flex justify-between">
              <span className="font-medium text-gray-900 dark:text-white">API Authentication Guide</span>
              <span className="text-sm text-gray-500 dark:text-gray-400">Updated 2 days ago</span>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">Added JWT token refresh workflow and examples for OAuth 2.0 integration.</p>
          </li>
          <li className="border-b pb-3 dark:border-gray-700">
            <div className="flex justify-between">
              <span className="font-medium text-gray-900 dark:text-white">Knowledge Base Management</span>
              <span className="text-sm text-gray-500 dark:text-gray-400">Updated 1 week ago</span>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">New section on document chunking strategies and vector embedding optimization.</p>
          </li>
          <li>
            <div className="flex justify-between">
              <span className="font-medium text-gray-900 dark:text-white">Agent-NN CLI Reference</span>
              <span className="text-sm text-gray-500 dark:text-gray-400">Updated 2 weeks ago</span>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">Updated with new commands for agent deployment and monitoring.</p>
          </li>
        </ul>
        
        <div className="mt-4 flex justify-end">
          <a href="#" className="flex items-center text-indigo-600 dark:text-indigo-400 hover:underline">
            View all documentation updates
            <ExternalLink size={14} className="ml-1" />
          </a>
        </div>
      </Card>
      
      {/* Support Section */}
      <Card title="Need Help?">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="border rounded-lg p-4 dark:border-gray-700">
            <h4 className="font-semibold mb-2 text-gray-900 dark:text-white">Community Forum</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">Ask questions and share knowledge with other Agent-NN users.</p>
            <a href="#" className="text-indigo-600 dark:text-indigo-400 hover:underline text-sm flex items-center">
              Visit forum
              <ExternalLink size={14} className="ml-1" />
            </a>
          </div>
          
          <div className="border rounded-lg p-4 dark:border-gray-700">
            <h4 className="font-semibold mb-2 text-gray-900 dark:text-white">GitHub Issues</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">Report bugs or suggest new features for Agent-NN.</p>
            <a href="#" className="text-indigo-600 dark:text-indigo-400 hover:underline text-sm flex items-center">
              Open an issue
              <ExternalLink size={14} className="ml-1" />
            </a>
          </div>
          
          <div className="border rounded-lg p-4 dark:border-gray-700">
            <h4 className="font-semibold mb-2 text-gray-900 dark:text-white">Professional Support</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">Get expert help for enterprise deployments.</p>
            <a href="#" className="text-indigo-600 dark:text-indigo-400 hover:underline text-sm flex items-center">
              Contact support
              <ExternalLink size={14} className="ml-1" />
            </a>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default DocsPanel;
