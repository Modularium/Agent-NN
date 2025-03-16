// monitoring/dashboard/components/panels/SecurityPanel.tsx
import React, { useState } from 'react';
import { Shield, Bell, AlertTriangle, Search, Filter, Lock, Key, User, Users, FileText, RefreshCw, Clock, ExternalLink } from 'lucide-react';
import { useDashboard } from '../../context/DashboardContext';
import Card from '../common/Card';
import StatusBadge from '../common/StatusBadge';

const SecurityPanel: React.FC = () => {
  const { securityStatus } = useDashboard();
  const [activeTab, setActiveTab] = useState<'overview' | 'events' | 'access' | 'settings'>('overview');
  const [timeRange, setTimeRange] = useState<string>('24h');
  
  // Mock security audit data
  const securityAudit = {
    lastAudit: '2025-03-10T08:00:00Z',
    nextAudit: '2025-04-10T08:00:00Z',
    score: 92,
    findings: [
      { severity: 'medium', description: 'API keys rotation policy not enforced', fixed: false },
      { severity: 'low', description: 'Password policy could be strengthened', fixed: true },
      { severity: 'info', description: 'Consider implementing MFA for admin accounts', fixed: false },
    ]
  };
  
  // Mock active sessions
  const activeSessions = [
    { id: 'session-1', user: 'admin', role: 'Administrator', ip: '192.168.1.105', startTime: '2025-03-16T08:30:00Z', lastActivity: '2025-03-16T14:15:00Z' },
    { id: 'session-2', user: 'john.doe', role: 'Manager', ip: '192.168.1.110', startTime: '2025-03-16T09:45:00Z', lastActivity: '2025-03-16T13:20:00Z' },
    { id: 'session-3', user: 'emma.wilson', role: 'User', ip: '192.168.1.120', startTime: '2025-03-16T10:15:00Z', lastActivity: '2025-03-16T11:30:00Z' },
  ];
  
  // Mock authentication methods
  const authMethods = [
    { name: 'JWT Token', enabled: true, default: true },
    { name: 'OAuth 2.0', enabled: true, default: false },
    { name: 'API Key', enabled: true, default: false },
    { name: 'LDAP', enabled: false, default: false },
  ];
  
  // Mock access roles
  const accessRoles = [
    { name: 'Administrator', users: 2, description: 'Full access to all features' },
    { name: 'Manager', users: 5, description: 'View and manage agents, knowledge bases' },
    { name: 'User', users: 28, description: 'Submit tasks and view results' },
    { name: 'Guest', users: 12, description: 'Read-only access to documentation' },
  ];
  
  // Function to format timestamp
  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };
  
  // Function to calculate time since
  const timeSince = (timestamp: string) => {
    const now = new Date();
    const date = new Date(timestamp);
    const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);
    
    let interval = seconds / 3600;
    if (interval > 24) {
      return Math.floor(interval / 24) + ' days ago';
    }
    if (interval > 1) {
      return Math.floor(interval) + ' hours ago';
    }
    interval = seconds / 60;
    if (interval > 1) {
      return Math.floor(interval) + ' minutes ago';
    }
    return Math.floor(seconds) + ' seconds ago';
  };
  
  // Function to get severity class
  const getSeverityClass = (severity: string): string => {
    switch (severity.toLowerCase()) {
      case 'high':
        return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300';
      case 'low':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300';
    }
  };
  
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">Security Management</h2>
        <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 flex items-center transition">
          <Shield size={18} className="mr-2" />
          Security Audit
        </button>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-gray-200 dark:border-gray-700">
        <button
          className={`px-4 py-2 font-medium ${
            activeTab === 'overview'
              ? 'border-b-2 border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('overview')}
        >
          Security Overview
        </button>
        <button
          className={`px-4 py-2 font-medium ${
            activeTab === 'events'
              ? 'border-b-2 border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('events')}
        >
          Security Events
        </button>
        <button
          className={`px-4 py-2 font-medium ${
            activeTab === 'access'
              ? 'border-b-2 border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('access')}
        >
          Access Control
        </button>
        <button
          className={`px-4 py-2 font-medium ${
            activeTab === 'settings'
              ? 'border-b-2 border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('settings')}
        >
          Security Settings
        </button>
      </div>

      {/* Security Overview Tab */}
      {activeTab === 'overview' && (
        <>
          <Card>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="border rounded-lg p-4 dark:border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold text-gray-900 dark:text-white">Security Status</h4>
                  <Shield size={18} className="text-green-600 dark:text-green-500" />
                </div>
                <StatusBadge status="Secure" />
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  Security systems are functioning normally.
                </p>
              </div>
              
              <div className="border rounded-lg p-4 dark:border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold text-gray-900 dark:text-white">Last Security Scan</h4>
                  <Shield size={18} className="text-indigo-600 dark:text-indigo-500" />
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {formatTimestamp(securityStatus?.lastScan || '')}
                </p>
                <p className="text-sm text-green-600 dark:text-green-500 mt-1">
                  No critical issues found
                </p>
              </div>
              
              <div className="border rounded-lg p-4 dark:border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold text-gray-900 dark:text-white">Detected Vulnerabilities</h4>
                  <Shield size={18} className="text-indigo-600 dark:text-indigo-500" />
                </div>
                <div className="grid grid-cols-3 gap-2">
                  <div className="flex flex-col items-center">
                    <span className="text-red-600 dark:text-red-500 font-bold text-xl">
                      {securityStatus?.vulnerabilities.high || 0}
                    </span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">High</span>
                  </div>
                  <div className="flex flex-col items-center">
                    <span className="text-yellow-600 dark:text-yellow-500 font-bold text-xl">
                      {securityStatus?.vulnerabilities.medium || 0}
                    </span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">Medium</span>
                  </div>
                  <div className="flex flex-col items-center">
                    <span className="text-blue-600 dark:text-blue-500 font-bold text-xl">
                      {securityStatus?.vulnerabilities.low || 0}
                    </span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">Low</span>
                  </div>
                </div>
              </div>
            </div>
          </Card>
          
          <Card title="Security Audit Results">
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Last audit: {formatTimestamp(securityAudit.lastAudit)}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Next scheduled audit: {formatTimestamp(securityAudit.nextAudit)}
                </p>
              </div>
              
              <div className="flex items-center">
                <div className="w-16 h-16 rounded-full border-4 border-green-500 flex items-center justify-center mr-4">
                  <span className="text-xl font-bold text-green-600 dark:text-green-500">{securityAudit.score}</span>
                </div>
                <div className="text-sm">
                  <p className="font-medium text-gray-900 dark:text-white">Security Score</p>
                  <p className="text-green-600 dark:text-green-500">Good</p>
                </div>
              </div>
            </div>
            
            <div className="border rounded-lg dark:border-gray-700 divide-y divide-gray-200 dark:divide-gray-700">
              {securityAudit.findings.map((finding, index) => (
                <div key={index} className="p-3 flex items-center justify-between">
                  <div className="flex items-start">
                    <span className={`inline-block w-2 h-2 rounded-full mt-1.5 mr-2 ${
                      finding.severity === 'medium' ? 'bg-yellow-500' : 
                      finding.severity === 'low' ? 'bg-blue-500' : 'bg-gray-500'
                    }`}></span>
                    <div>
                      <p className="text-sm font-medium text-gray-900 dark:text-white">{finding.description}</p>
                      <span className={`text-xs mt-1 inline

<span className={`text-xs mt-1 inline-flex items-center ${
                        finding.severity === 'medium' ? 'text-yellow-700 dark:text-yellow-500' : 
                        finding.severity === 'low' ? 'text-blue-700 dark:text-blue-500' : 'text-gray-700 dark:text-gray-500'
                      }`}>
                        {finding.severity.charAt(0).toUpperCase() + finding.severity.slice(1)} severity
                      </span>
                    </div>
                  </div>
                  <div>
                    {finding.fixed ? (
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300">
                        Fixed
                      </span>
                    ) : (
                      <button className="text-xs px-2 py-1 bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-400 rounded hover:bg-indigo-200 dark:hover:bg-indigo-900/50">
                        Fix Issue
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
            
            <div className="mt-4 flex justify-end">
              <a href="#" className="text-indigo-600 dark:text-indigo-400 hover:underline text-sm flex items-center">
                View full audit report
                <ExternalLink size={14} className="ml-1" />
              </a>
            </div>
          </Card>
          
          <Card title="Active Sessions">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                <thead>
                  <tr>
                    <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">User</th>
                    <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Role</th>
                    <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">IP Address</th>
                    <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Session Start</th>
                    <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Last Activity</th>
                    <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {activeSessions.map((session) => (
                    <tr key={session.id} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                      <td className="px-3 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">{session.user}</td>
                      <td className="px-3 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{session.role}</td>
                      <td className="px-3 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{session.ip}</td>
                      <td className="px-3 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{formatTimestamp(session.startTime)}</td>
                      <td className="px-3 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{timeSince(session.lastActivity)}</td>
                      <td className="px-3 py-4 whitespace-nowrap text-sm">
                        <button className="text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300">
                          Terminate
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* Security Events Tab */}
      {activeTab === 'events' && (
        <Card>
          <div className="flex justify-between items-center mb-4">
            <div className="flex items-center space-x-2">
              <div className="relative">
                <input
                  type="text"
                  className="pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white rounded-md"
                  placeholder="Search security events..."
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
            <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 flex items-center transition">
              Export Events
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead>
                <tr>
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Event Type</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Timestamp</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Severity</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Details</th>
                  <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {securityStatus?.events.map((event, index) => (
                  <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                    <td className="px-3 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">{event.type}</td>
                    <td className="px-3 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      <div className="flex items-center">
                        <Clock size={14} className="mr-1" />
                        {formatTimestamp(event.timestamp)}
                      </div>
                    </td>
                    <td className="px-3 py-4 whitespace-nowrap text-sm">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityClass(event.severity)}`}>
                        {event.severity}
                      </span>
                    </td>
                    <td className="px-3 py-4 text-sm text-gray-900 dark:text-white">{event.details}</td>
                    <td className="px-3 py-4 whitespace-nowrap text-sm">
                      <button className="px-2 py-1 bg-indigo-100 text-indigo-700 rounded text-xs dark:bg-indigo-900/30 dark:text-indigo-400">
                        View Details
                      </button>
                    </td>
                  </tr>
                ))}
                {/* Additional mock entries to fill the table */}
                <tr className="hover:bg-gray-50 dark:hover:bg-gray-800">
                  <td className="px-3 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">Failed Login</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    <div className="flex items-center">
                      <Clock size={14} className="mr-1" />
                      {formatTimestamp(new Date(Date.now() - 30 * 60000).toISOString())}
                    </div>
                  </td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityClass('medium')}`}>
                      medium
                    </span>
                  </td>
                  <td className="px-3 py-4 text-sm text-gray-900 dark:text-white">Failed login attempt: user 'john.doe' from 192.168.1.150</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">
                    <button className="px-2 py-1 bg-indigo-100 text-indigo-700 rounded text-xs dark:bg-indigo-900/30 dark:text-indigo-400">
                      View Details
                    </button>
                  </td>
                </tr>
                <tr className="hover:bg-gray-50 dark:hover:bg-gray-800">
                  <td className="px-3 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">Config Change</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    <div className="flex items-center">
                      <Clock size={14} className="mr-1" />
                      {formatTimestamp(new Date(Date.now() - 120 * 60000).toISOString())}
                    </div>
                  </td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityClass('low')}`}>
                      low
                    </span>
                  </td>
                  <td className="px-3 py-4 text-sm text-gray-900 dark:text-white">Security settings updated by admin</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">
                    <button className="px-2 py-1 bg-indigo-100 text-indigo-700 rounded text-xs dark:bg-indigo-900/30 dark:text-indigo-400">
                      View Details
                    </button>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="flex justify-between items-center mt-4">
            <div className="text-sm text-gray-500 dark:text-gray-400">
              Showing 5 of 248 events
            </div>
            <div className="flex space-x-2">
              <button className="px-3 py-1 bg-gray-100 dark:bg-gray-800 rounded-md text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700">
                Previous
              </button>
              <button className="px-3 py-1 bg-indigo-100 dark:bg-indigo-900/30 rounded-md text-sm text-indigo-700 dark:text-indigo-400">
                1
              </button>
              <button className="px-3 py-1 bg-gray-100 dark:bg-gray-800 rounded-md text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700">
                2
              </button>
              <button className="px-3 py-1 bg-gray-100 dark:bg-gray-800 rounded-md text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700">
                3
              </button>
              <button className="px-3 py-1 bg-gray-100 dark:bg-gray-800 rounded-md text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700">
                Next
              </button>
            </div>
          </div>
        </Card>
      )}

      {/* Access Control Tab */}
      {activeTab === 'access' && (
        <>
          <Card title="Authentication Methods">
            <div className="mb-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {authMethods.map((method, index) => (
                  <div key={index} className="border rounded-lg p-4 dark:border-gray-700">
                    <div className="flex justify-between items-center">
                      <div>
                        <h4 className="font-medium text-gray-900 dark:text-white">{method.name}</h4>
                        {method.default && (
                          <span className="text-xs text-indigo-600 dark:text-indigo-400">Default method</span>
                        )}
                      </div>
                      <div className="relative inline-block w-12 h-6 border rounded-full dark:border-gray-700">
                        <input type="checkbox" className="sr-only" defaultChecked={method.enabled} />
                        <span 
                          className={`block absolute left-1 top-1 w-4 h-4 rounded-full transition-transform ${
                            method.enabled 
                              ? 'bg-indigo-600 dark:bg-indigo-500 transform translate-x-6' 
                              : 'bg-gray-300 dark:bg-gray-600'
                          }`}
                        ></span>
                      </div>
                    </div>
                    {method.name === 'JWT Token' && (
                      <div className="mt-2 space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600 dark:text-gray-400">Token expiry:</span>
                          <span className="text-gray-900 dark:text-white">24 hours</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600 dark:text-gray-400">Refresh tokens:</span>
                          <span className="text-gray-900 dark:text-white">Enabled</span>
                        </div>
                      </div>
                    )}
                    {method.name === 'OAuth 2.0' && (
                      <div className="mt-2 space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600 dark:text-gray-400">Providers:</span>
                          <span className="text-gray-900 dark:text-white">Google, GitHub</span>
                        </div>
                      </div>
                    )}
                    {method.name === 'API Key' && (
                      <div className="mt-2 space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600 dark:text-gray-400">Key rotation:</span>
                          <span className="text-gray-900 dark:text-white">90 days</span>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
              <div className="flex justify-end mt-4">
                <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 transition">
                  Configure Authentication
                </button>
              </div>
            </div>
          </Card>
          
          <Card title="Access Roles">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                <thead>
                  <tr>
                    <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Role</th>
                    <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Description</th>
                    <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Users</th>
                    <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Permissions</th>
                    <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {accessRoles.map((role, index) => (
                    <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                      <td className="px-3 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">{role.name}</td>
                      <td className="px-3 py-4 text-sm text-gray-500 dark:text-gray-400">{role.description}</td>
                      <td className="px-3 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{role.users}</td>
                      <td className="px-3 py-4 whitespace-nowrap text-sm">
                        <button className="px-2 py-1 bg-indigo-100 text-indigo-700 rounded text-xs dark:bg-indigo-900/30 dark:text-indigo-400">
                          View Permissions
                        </button>
                      </td>
                      <td className="px-3 py-4 whitespace-nowrap text-sm">
                        <div className="flex space-x-2">
                          <button className="text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-300">Edit</button>
                          {role.name !== 'Administrator' && (
                            <button className="text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300">Delete</button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            <div className="flex justify-between items-center mt-4">
              <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 transition flex items-center">
                <Users size={16} className="mr-2" />
                Manage Users
              </button>
              <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 transition flex items-center">
                <Plus size={16} className="mr-2" />
                Create New Role
              </button>
            </div>
          </Card>
        </>
      )}

      {/* Security Settings Tab */}
      {activeTab === 'settings' && (
        <Card title="Security Settings">
          <div className="space-y-6">
            <div>
              <h3 className="font-medium text-gray-900 dark:text-white mb-3">Password Policy</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Minimum Password Length</label>
                  <div className="flex items-center">
                    <input type="number" className="form-input rounded-md w-24 border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white" defaultValue="12" />
                    <span className="ml-2 text-sm text-gray-500 dark:text-gray-400">characters</span>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Password Expiry</label>
                  <div className="flex items-center">
                    <input type="number" className="form-input rounded-md w-24 border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white" defaultValue="90" />
                    <span className="ml-2 text-sm text-gray-500 dark:text-gray-400">days</span>
                  </div>
                </div>
              </div>
              
              <div className="mt-3 space-y-2">
                <div className="flex items-center">
                  <input type="checkbox" className="mr-2" defaultChecked />
                  <span className="text-gray-700 dark:text-gray-300">Require uppercase letters</span>
                </div>
                <div className="flex items-center">
                  <input type="checkbox" className="mr-2" defaultChecked />
                  <span className="text-gray-700 dark:text-gray-300">Require lowercase letters</span>
                </div>
                <div className="flex items-center">
                  <input type="checkbox" className="mr-2" defaultChecked />
                  <span className="text-gray-700 dark:text-gray-300">Require numbers</span>
                </div>
                <div className="flex items-center">
                  <input type="checkbox" className="mr-2" defaultChecked />
                  <span className="text-gray-700 dark:text-gray-300">Require special characters</span>
                </div>
                <div className="flex items-center">
                  <input type="checkbox" className="mr-2" defaultChecked />
                  <span className="text-gray-700 dark:text-gray-300">Prevent password reuse (last 5 passwords)</span>
                </div>
              </div>
            </div>
            
            <div className="pt-4 border-t dark:border-gray-700">
              <h3 className="font-medium text-gray-900 dark:text-white mb-3">Multi-Factor Authentication</h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium text-gray-700 dark:text-gray-300">Require MFA for all users</h4>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      Force all users to use multi-factor authentication
                    </p>
                  </div>
                  <div className="relative inline-block w-12 h-6 border rounded-full dark:border-gray-700">
                    <input type="checkbox" className="sr-only" />
                    <span className="block absolute left-1 top-1 bg-gray-300 dark:bg-gray-600 w-4 h-4 rounded-full transition-transform"></span>
                  </div>
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium text-gray-700 dark:text-gray-300">Require MFA for administrators</h4>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      Force administrators to use multi-factor authentication
                    </p>
                  </div>
                  <div className="relative inline-block w-12 h-6 border rounded-full dark:border-gray-700">
                    <input type="checkbox" className="sr-only" defaultChecked />
                    <span className="block absolute left-1 top-1 bg-indigo-600 dark:bg-indigo-500 w-4 h-4 rounded-full transition-transform transform translate-x-6"></span>
                  </div>
                </div>
              </div>
              
              <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="flex items-center space-x-2 border rounded-lg p-3 dark:border-gray-700">
                  <input type="checkbox" defaultChecked />
                  <div>
                    <h4 className="font-medium text-gray-700 dark:text-gray-300">Authenticator Apps</h4>
                    <p className="text-xs text-gray-500 dark:text-gray-400">Google, Microsoft, etc.</p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2 border rounded-lg p-3 dark:border-gray-700">
                  <input type="checkbox" defaultChecked />
                  <div>
                    <h4 className="font-medium text-gray-700 dark:text-gray-300">SMS Authentication</h4>
                    <p className="text-xs text-gray-500 dark:text-gray-400">Text message codes</p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2 border rounded-lg p-3 dark:border-gray-700">
                  <input type="checkbox" />
                  <div>
                    <h4 className="font-medium text-gray-700 dark:text-gray-300">Hardware Keys</h4>
                    <p className="text-xs text-gray-500 dark:text-gray-400">YubiKey, etc.</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="flex space-x-3 pt-4 border-t dark:border-gray-700 mt-4">
              <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 transition">
                Save Settings
              </button>
              <button className="px-4 py-2 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600 transition">
                Reset to Defaults
              </button>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default SecurityPanel;// monitoring/dashboard/components/panels/SecurityPanel.tsx
import React, { useState } from 'react';
import { Shield, Bell, AlertTriangle, Search, Filter, Lock, Key, User, Users, FileText, RefreshCw, Clock, ExternalLink } from 'lucide-react';
import { useDashboard } from '../../context/DashboardContext';
import Card from '../common/Card';
import StatusBadge from '../common/StatusBadge';

const SecurityPanel: React.FC = () => {
  const { securityStatus } = useDashboard();
  const [activeTab, setActiveTab] = useState<'overview' | 'events' | 'access' | 'settings'>('overview');
  const [timeRange, setTimeRange] = useState<string>('24h');
  
  // Mock security audit data
  const securityAudit = {
    lastAudit: '2025-03-10T08:00:00Z',
    nextAudit: '2025-04-10T08:00:00Z',
    score: 92,
    findings: [
      { severity: 'medium', description: 'API keys rotation policy not enforced', fixed: false },
      { severity: 'low', description: 'Password policy could be strengthened', fixed: true },
      { severity: 'info', description: 'Consider implementing MFA for admin accounts', fixed: false },
    ]
  };
  
  // Mock active sessions
  const activeSessions = [
    { id: 'session-1', user: 'admin', role: 'Administrator', ip: '192.168.1.105', startTime: '2025-03-16T08:30:00Z', lastActivity: '2025-03-16T14:15:00Z' },
    { id: 'session-2', user: 'john.doe', role: 'Manager', ip: '192.168.1.110', startTime: '2025-03-16T09:45:00Z', lastActivity: '2025-03-16T13:20:00Z' },
    { id: 'session-3', user: 'emma.wilson', role: 'User', ip: '192.168.1.120', startTime: '2025-03-16T10:15:00Z', lastActivity: '2025-03-16T11:30:00Z' },
  ];
  
  // Mock authentication methods
  const authMethods = [
    { name: 'JWT Token', enabled: true, default: true },
    { name: 'OAuth 2.0', enabled: true, default: false },
    { name: 'API Key', enabled: true, default: false },
    { name: 'LDAP', enabled: false, default: false },
  ];
  
  // Mock access roles
  const accessRoles = [
    { name: 'Administrator', users: 2, description: 'Full access to all features' },
    { name: 'Manager', users: 5, description: 'View and manage agents, knowledge bases' },
    { name: 'User', users: 28, description: 'Submit tasks and view results' },
    { name: 'Guest', users: 12, description: 'Read-only access to documentation' },
  ];
  
  // Function to format timestamp
  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };
  
  // Function to calculate time since
  const timeSince = (timestamp: string) => {
    const now = new Date();
    const date = new Date(timestamp);
    const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);
    
    let interval = seconds / 3600;
    if (interval > 24) {
      return Math.floor(interval / 24) + ' days ago';
    }
    if (interval > 1) {
      return Math.floor(interval) + ' hours ago';
    }
    interval = seconds / 60;
    if (interval > 1) {
      return Math.floor(interval) + ' minutes ago';
    }
    return Math.floor(seconds) + ' seconds ago';
  };
  
  // Function to get severity class
  const getSeverityClass = (severity: string): string => {
    switch (severity.toLowerCase()) {
      case 'high':
        return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300';
      case 'low':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300';
    }
  };
  
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">Security Management</h2>
        <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 flex items-center transition">
          <Shield size={18} className="mr-2" />
          Security Audit
        </button>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-gray-200 dark:border-gray-700">
        <button
          className={`px-4 py-2 font-medium ${
            activeTab === 'overview'
              ? 'border-b-2 border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('overview')}
        >
          Security Overview
        </button>
        <button
          className={`px-4 py-2 font-medium ${
            activeTab === 'events'
              ? 'border-b-2 border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('events')}
        >
          Security Events
        </button>
        <button
          className={`px-4 py-2 font-medium ${
            activeTab === 'access'
              ? 'border-b-2 border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('access')}
        >
          Access Control
        </button>
        <button
          className={`px-4 py-2 font-medium ${
            activeTab === 'settings'
              ? 'border-b-2 border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('settings')}
        >
          Security Settings
        </button>
      </div>

      {/* Security Overview Tab */}
      {activeTab === 'overview' && (
        <>
          <Card>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="border rounded-lg p-4 dark:border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold text-gray-900 dark:text-white">Security Status</h4>
                  <Shield size={18} className="text-green-600 dark:text-green-500" />
                </div>
                <StatusBadge status="Secure" />
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  Security systems are functioning normally.
                </p>
              </div>
              
              <div className="border rounded-lg p-4 dark:border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold text-gray-900 dark:text-white">Last Security Scan</h4>
                  <Shield size={18} className="text-indigo-600 dark:text-indigo-500" />
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {formatTimestamp(securityStatus?.lastScan || '')}
                </p>
                <p className="text-sm text-green-600 dark:text-green-500 mt-1">
                  No critical issues found
                </p>
              </div>
              
              <div className="border rounded-lg p-4 dark:border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold text-gray-900 dark:text-white">Detected Vulnerabilities</h4>
                  <Shield size={18} className="text-indigo-600 dark:text-indigo-500" />
                </div>
                <div className="grid grid-cols-3 gap-2">
                  <div className="flex flex-col items-center">
                    <span className="text-red-600 dark:text-red-500 font-bold text-xl">
                      {securityStatus?.vulnerabilities.high || 0}
                    </span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">High</span>
                  </div>
                  <div className="flex flex-col items-center">
                    <span className="text-yellow-600 dark:text-yellow-500 font-bold text-xl">
                      {securityStatus?.vulnerabilities.medium || 0}
                    </span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">Medium</span>
                  </div>
                  <div className="flex flex-col items-center">
                    <span className="text-blue-600 dark:text-blue-500 font-bold text-xl">
                      {securityStatus?.vulnerabilities.low || 0}
                    </span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">Low</span>
                  </div>
                </div>
              </div>
            </div>
          </Card>
          
          <Card title="Security Audit Results">
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Last audit: {formatTimestamp(securityAudit.lastAudit)}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Next scheduled audit: {formatTimestamp(securityAudit.nextAudit)}
                </p>
              </div>
              
              <div className="flex items-center">
                <div className="w-16 h-16 rounded-full border-4 border-green-500 flex items-center justify-center mr-4">
                  <span className="text-xl font-bold text-green-600 dark:text-green-500">{securityAudit.score}</span>
                </div>
                <div className="text-sm">
                  <p className="font-medium text-gray-900 dark:text-white">Security Score</p>
                  <p className="text-green-600 dark:text-green-500">Good</p>
                </div>
              </div>
            </div>
            
            <div className="border rounded-lg dark:border-gray-700 divide-y divide-gray-200 dark:divide-gray-700">
              {securityAudit.findings.map((finding, index) => (
                <div key={index} className="p-3 flex items-center justify-between">
                  <div className="flex items-start">
                    <span className={`inline-block w-2 h-2 rounded-full mt-1.5 mr-2 ${
                      finding.severity === 'medium' ? 'bg-yellow-500' : 
                      finding.severity === 'low' ? 'bg-blue-500' : 'bg-gray-500'
                    }`}></span>
                    <div>
                      <p className="text-sm font-medium text-gray-900 dark:text-white">{finding.description}</p>
                      <span className={`text-xs mt-1 inline
