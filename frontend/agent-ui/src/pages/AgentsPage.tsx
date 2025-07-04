import { useState, useEffect } from 'react'

interface Agent {
  id: string
  name: string
  domain: string
  totalTasks: number
  successRate: number
  status: 'active' | 'idle' | 'error'
  lastActive: Date
  description: string
  version: string
}

const mockAgents: Agent[] = [
  {
    id: '1',
    name: 'DockerMaster',
    domain: 'Container Management',
    totalTasks: 247,
    successRate: 0.96,
    status: 'active',
    lastActive: new Date(),
    description: 'Specialized in Docker container operations and orchestration',
    version: 'v2.1.0'
  },
  {
    id: '2',
    name: 'CodeAnalyzer',
    domain: 'Code Analysis',
    totalTasks: 156,
    successRate: 0.92,
    status: 'idle',
    lastActive: new Date(Date.now() - 300000),
    description: 'Advanced code review and analysis capabilities',
    version: 'v1.8.2'
  },
  {
    id: '3',
    name: 'DataProcessor',
    domain: 'Data Processing',
    totalTasks: 89,
    successRate: 0.88,
    status: 'active',
    lastActive: new Date(Date.now() - 120000),
    description: 'Handles complex data transformation and analysis tasks',
    version: 'v3.0.1'
  },
  {
    id: '4',
    name: 'NetworkGuard',
    domain: 'Security',
    totalTasks: 312,
    successRate: 0.99,
    status: 'error',
    lastActive: new Date(Date.now() - 600000),
    description: 'Network security monitoring and threat detection',
    version: 'v1.5.3'
  },
]

export default function ModernAgentsPage() {
  const [agents, setAgents] = useState<Agent[]>([])
  const [viewMode, setViewMode] = useState<'grid' | 'table'>('grid')
  const [sortBy, setSortBy] = useState<'name' | 'tasks' | 'success'>('name')
  const [filterStatus, setFilterStatus] = useState<'all' | 'active' | 'idle' | 'error'>('all')
  const [searchTerm, setSearchTerm] = useState('')

  useEffect(() => {
    // Simulate API call
    setTimeout(() => setAgents(mockAgents), 500)
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800 border-green-200'
      case 'idle': return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      case 'error': return 'bg-red-100 text-red-800 border-red-200'
      default: return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return '🟢'
      case 'idle': return '🟡'
      case 'error': return '🔴'
      default: return '⚪'
    }
  }

  const filteredAgents = agents
    .filter(agent => 
      (filterStatus === 'all' || agent.status === filterStatus) &&
      (searchTerm === '' || agent.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
       agent.domain.toLowerCase().includes(searchTerm.toLowerCase()))
    )
    .sort((a, b) => {
      switch (sortBy) {
        case 'tasks': return b.totalTasks - a.totalTasks
        case 'success': return b.successRate - a.successRate
        default: return a.name.localeCompare(b.name)
      }
    })

  const formatLastActive = (date: Date) => {
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const minutes = Math.floor(diff / 60000)
    
    if (minutes < 1) return 'Just now'
    if (minutes < 60) return `${minutes}m ago`
    if (minutes < 1440) return `${Math.floor(minutes / 60)}h ago`
    return `${Math.floor(minutes / 1440)}d ago`
  }

  const AgentCard = ({ agent }: { agent: Agent }) => (
    <div className="bg-white rounded-xl border border-gray-200 p-6 hover:shadow-lg transition-all duration-200 group">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-md">
            <span className="text-white font-bold text-lg">
              {agent.name.charAt(0)}
            </span>
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 group-hover:text-blue-600 transition-colors">
              {agent.name}
            </h3>
            <p className="text-sm text-gray-500">{agent.version}</p>
          </div>
        </div>
        <span className={`px-2 py-1 text-xs rounded-full border ${getStatusColor(agent.status)}`}>
          {getStatusIcon(agent.status)} {agent.status}
        </span>
      </div>

      <p className="text-gray-600 text-sm mb-4 line-clamp-2">
        {agent.description}
      </p>

      <div className="space-y-3">
        <div className="flex justify-between text-sm">
          <span className="text-gray-500">Domain:</span>
          <span className="font-medium text-gray-900">{agent.domain}</span>
        </div>
        
        <div className="flex justify-between text-sm">
          <span className="text-gray-500">Tasks:</span>
          <span className="font-medium text-gray-900">{agent.totalTasks}</span>
        </div>

        <div className="flex justify-between text-sm items-center">
          <span className="text-gray-500">Success Rate:</span>
          <div className="flex items-center gap-2">
            <div className="w-16 bg-gray-200 rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-green-500 to-emerald-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${agent.successRate * 100}%` }}
              ></div>
            </div>
            <span className="font-medium text-gray-900">{(agent.successRate * 100).toFixed(1)}%</span>
          </div>
        </div>

        <div className="flex justify-between text-sm">
          <span className="text-gray-500">Last Active:</span>
          <span className="font-medium text-gray-900">{formatLastActive(agent.lastActive)}</span>
        </div>
      </div>

      <div className="mt-4 pt-4 border-t border-gray-100 flex gap-2">
        <button className="flex-1 px-3 py-2 text-sm bg-blue-50 text-blue-600 rounded-lg hover:bg-blue-100 transition-colors">
          Configure
        </button>
        <button className="flex-1 px-3 py-2 text-sm bg-gray-50 text-gray-600 rounded-lg hover:bg-gray-100 transition-colors">
          Monitor
        </button>
      </div>
    </div>
  )

  const AgentTableRow = ({ agent }: { agent: Agent }) => (
    <tr className="border-b border-gray-200 hover:bg-gray-50 transition-colors">
      <td className="p-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">
              {agent.name.charAt(0)}
            </span>
          </div>
          <div>
            <div className="font-medium text-gray-900">{agent.name}</div>
            <div className="text-sm text-gray-500">{agent.version}</div>
          </div>
        </div>
      </td>
      <td className="p-4">
        <span className="text-gray-900">{agent.domain}</span>
      </td>
      <td className="p-4">
        <span className="font-medium">{agent.totalTasks}</span>
      </td>
      <td className="p-4">
        <div className="flex items-center gap-2">
          <div className="w-12 bg-gray-200 rounded-full h-2">
            <div 
              className="bg-gradient-to-r from-green-500 to-emerald-600 h-2 rounded-full"
              style={{ width: `${agent.successRate * 100}%` }}
            ></div>
          </div>
          <span className="text-sm font-medium">{(agent.successRate * 100).toFixed(1)}%</span>
        </div>
      </td>
      <td className="p-4">
        <span className={`px-2 py-1 text-xs rounded-full border ${getStatusColor(agent.status)}`}>
          {getStatusIcon(agent.status)} {agent.status}
        </span>
      </td>
      <td className="p-4 text-sm text-gray-600">
        {formatLastActive(agent.lastActive)}
      </td>
      <td className="p-4">
        <div className="flex gap-1">
          <button className="p-1 text-gray-400 hover:text-blue-600 transition-colors">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
            </svg>
          </button>
          <button className="p-1 text-gray-400 hover:text-green-600 transition-colors">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </button>
          <button className="p-1 text-gray-400 hover:text-red-600 transition-colors">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </button>
        </div>
      </td>
    </tr>
  )

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Agents</h1>
          <p className="text-gray-600">Manage and monitor your AI agents</p>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                <span className="text-blue-600">🤖</span>
              </div>
              <span className="text-sm font-medium text-gray-600">Total Agents</span>
            </div>
            <p className="text-2xl font-bold text-gray-900">{agents.length}</p>
          </div>

          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                <span className="text-green-600">🟢</span>
              </div>
              <span className="text-sm font-medium text-gray-600">Active</span>
            </div>
            <p className="text-2xl font-bold text-gray-900">
              {agents.filter(a => a.status === 'active').length}
            </p>
          </div>

          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 bg-yellow-100 rounded-lg flex items-center justify-center">
                <span className="text-yellow-600">📊</span>
              </div>
              <span className="text-sm font-medium text-gray-600">Avg Success Rate</span>
            </div>
            <p className="text-2xl font-bold text-gray-900">
              {agents.length > 0 ? 
                ((agents.reduce((acc, a) => acc + a.successRate, 0) / agents.length) * 100).toFixed(1) + '%' 
                : '0%'
              }
            </p>
          </div>

          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center">
                <span className="text-purple-600">📋</span>
              </div>
              <span className="text-sm font-medium text-gray-600">Total Tasks</span>
            </div>
            <p className="text-2xl font-bold text-gray-900">
              {agents.reduce((acc, a) => acc + a.totalTasks, 0)}
            </p>
          </div>
        </div>

        {/* Controls */}
        <div className="bg-white rounded-xl border border-gray-200 p-6 mb-6">
          <div className="flex flex-col lg:flex-row gap-4 items-start lg:items-center justify-between">
            <div className="flex flex-col sm:flex-row gap-4 flex-1">
              <div className="relative">
                <input
                  type="text"
                  placeholder="Search agents..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:shadow-focus focus:border-transparent"
                />
                <svg className="w-5 h-5 text-gray-400 absolute left-3 top-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>

              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value as any)}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:shadow-focus focus:border-transparent"
              >
                <option value="all">All Status</option>
                <option value="active">Active</option>
                <option value="idle">Idle</option>
                <option value="error">Error</option>
              </select>

              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as any)}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:shadow-focus focus:border-transparent"
              >
                <option value="name">Sort by Name</option>
                <option value="tasks">Sort by Tasks</option>
                <option value="success">Sort by Success Rate</option>
              </select>
            </div>

            <div className="flex gap-2">
              <button
                onClick={() => setViewMode('grid')}
                className={`p-2 rounded-lg transition-colors ${
                  viewMode === 'grid' 
                    ? 'bg-blue-100 text-blue-600' 
                    : 'text-gray-400 hover:text-gray-600'
                }`}
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                </svg>
              </button>
              <button
                onClick={() => setViewMode('table')}
                className={`p-2 rounded-lg transition-colors ${
                  viewMode === 'table' 
                    ? 'bg-blue-100 text-blue-600' 
                    : 'text-gray-400 hover:text-gray-600'
                }`}
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
                </svg>
              </button>
            </div>
          </div>
        </div>

        {/* Content */}
        {filteredAgents.length === 0 ? (
          <div className="bg-white rounded-xl border border-gray-200 p-12 text-center">
            <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-gray-400 text-2xl">🤖</span>
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">No agents found</h3>
            <p className="text-gray-500">Try adjusting your search or filter criteria</p>
          </div>
        ) : viewMode === 'grid' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredAgents.map((agent) => (
              <AgentCard key={agent.id} agent={agent} />
            ))}
          </div>
        ) : (
          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="text-left p-4 font-medium text-gray-900">Agent</th>
                  <th className="text-left p-4 font-medium text-gray-900">Domain</th>
                  <th className="text-left p-4 font-medium text-gray-900">Tasks</th>
                  <th className="text-left p-4 font-medium text-gray-900">Success Rate</th>
                  <th className="text-left p-4 font-medium text-gray-900">Status</th>
                  <th className="text-left p-4 font-medium text-gray-900">Last Active</th>
                  <th className="text-left p-4 font-medium text-gray-900">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredAgents.map((agent) => (
                  <AgentTableRow key={agent.id} agent={agent} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
