import { useState, useEffect } from 'react'

interface RoutingRule {
  id: string
  name: string
  description: string
  enabled: boolean
  priority: number
  conditions: {
    taskType?: string[]
    keywords?: string[]
    agentDomain?: string[]
    priority?: string[]
    contentLength?: { min?: number; max?: number }
  }
  target: {
    type: 'agent' | 'model' | 'custom'
    value: string
    fallback?: string
  }
  stats: {
    totalRequests: number
    successRate: number
    avgResponseTime: number
    lastUsed?: Date
  }
}

interface Model {
  id: string
  name: string
  provider: string
  version: string
  status: 'available' | 'unavailable' | 'maintenance'
  capabilities: string[]
  pricing: {
    inputTokens: number
    outputTokens: number
    currency: string
  }
  limits: {
    maxTokens: number
    requestsPerMinute: number
  }
  stats: {
    avgResponseTime: number
    successRate: number
    totalRequests: number
  }
}

const mockRoutingRules: RoutingRule[] = [
  {
    id: '1',
    name: 'Docker Tasks → DockerMaster',
    description: 'Route all Docker-related tasks to specialized Docker agent',
    enabled: true,
    priority: 1,
    conditions: {
      taskType: ['docker', 'container'],
      keywords: ['docker', 'container', 'deployment'],
      agentDomain: ['Container Management']
    },
    target: {
      type: 'agent',
      value: 'DockerMaster',
      fallback: 'GeneralAgent'
    },
    stats: {
      totalRequests: 247,
      successRate: 96.8,
      avgResponseTime: 2.3,
      lastUsed: new Date(Date.now() - 120000)
    }
  },
  {
    id: '2',
    name: 'High Priority → GPT-4',
    description: 'Route high priority tasks to most capable model',
    enabled: true,
    priority: 2,
    conditions: {
      priority: ['high', 'urgent']
    },
    target: {
      type: 'model',
      value: 'gpt-4-turbo',
      fallback: 'gpt-3.5-turbo'
    },
    stats: {
      totalRequests: 89,
      successRate: 98.9,
      avgResponseTime: 4.1,
      lastUsed: new Date(Date.now() - 300000)
    }
  },
  {
    id: '3',
    name: 'Long Content → Claude',
    description: 'Route long content analysis to Claude models',
    enabled: true,
    priority: 3,
    conditions: {
      contentLength: { min: 5000 },
      taskType: ['analysis', 'processing']
    },
    target: {
      type: 'model',
      value: 'claude-3-opus',
      fallback: 'claude-3-sonnet'
    },
    stats: {
      totalRequests: 156,
      successRate: 94.2,
      avgResponseTime: 6.7,
      lastUsed: new Date(Date.now() - 450000)
    }
  }
]

const mockModels: Model[] = [
  {
    id: 'gpt-4-turbo',
    name: 'GPT-4 Turbo',
    provider: 'OpenAI',
    version: '2024-04-09',
    status: 'available',
    capabilities: ['reasoning', 'coding', 'analysis', 'creative'],
    pricing: { inputTokens: 0.01, outputTokens: 0.03, currency: 'USD' },
    limits: { maxTokens: 128000, requestsPerMinute: 100 },
    stats: { avgResponseTime: 4.2, successRate: 98.5, totalRequests: 1247 }
  },
  {
    id: 'gpt-3.5-turbo',
    name: 'GPT-3.5 Turbo',
    provider: 'OpenAI',
    version: '0125',
    status: 'available',
    capabilities: ['general', 'coding', 'analysis'],
    pricing: { inputTokens: 0.0005, outputTokens: 0.0015, currency: 'USD' },
    limits: { maxTokens: 16385, requestsPerMinute: 500 },
    stats: { avgResponseTime: 1.8, successRate: 97.2, totalRequests: 3456 }
  },
  {
    id: 'claude-3-opus',
    name: 'Claude 3 Opus',
    provider: 'Anthropic',
    version: '20240229',
    status: 'available',
    capabilities: ['reasoning', 'analysis', 'creative', 'long-context'],
    pricing: { inputTokens: 0.015, outputTokens: 0.075, currency: 'USD' },
    limits: { maxTokens: 200000, requestsPerMinute: 50 },
    stats: { avgResponseTime: 6.1, successRate: 96.8, totalRequests: 892 }
  },
  {
    id: 'local-llama',
    name: 'Local Llama 2',
    provider: 'Local',
    version: '13B',
    status: 'maintenance',
    capabilities: ['general', 'coding'],
    pricing: { inputTokens: 0, outputTokens: 0, currency: 'USD' },
    limits: { maxTokens: 4096, requestsPerMinute: 10 },
    stats: { avgResponseTime: 12.5, successRate: 89.3, totalRequests: 234 }
  }
]

export default function ModernRoutingPage() {
  const [routingRules, setRoutingRules] = useState<RoutingRule[]>(mockRoutingRules)
  const [models, setModels] = useState<Model[]>(mockModels)
  const [activeTab, setActiveTab] = useState<'rules' | 'models' | 'analytics'>('rules')
  const [selectedRule, setSelectedRule] = useState<string | null>(null)
  const [isCreatingRule, setIsCreatingRule] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState<'all' | 'enabled' | 'disabled'>('all')

  const getStatusColor = (enabled: boolean) => {
    return enabled 
      ? 'bg-green-100 text-green-800 border-green-200'
      : 'bg-gray-100 text-gray-800 border-gray-200'
  }

  const getModelStatusColor = (status: string) => {
    switch (status) {
      case 'available': return 'bg-green-100 text-green-800 border-green-200'
      case 'maintenance': return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      case 'unavailable': return 'bg-red-100 text-red-800 border-red-200'
      default: return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  const formatCurrency = (amount: number, currency: string) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency,
      minimumFractionDigits: 4
    }).format(amount)
  }

  const formatTimeAgo = (date: Date) => {
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const minutes = Math.floor(diff / 60000)
    
    if (minutes < 1) return 'Just now'
    if (minutes < 60) return `${minutes}m ago`
    if (minutes < 1440) return `${Math.floor(minutes / 60)}h ago`
    return `${Math.floor(minutes / 1440)}d ago`
  }

  const toggleRule = (id: string) => {
    setRoutingRules(prev => prev.map(rule => 
      rule.id === id ? { ...rule, enabled: !rule.enabled } : rule
    ))
  }

  const filteredRules = routingRules.filter(rule => {
    const matchesSearch = rule.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         rule.description.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesStatus = statusFilter === 'all' || 
                         (statusFilter === 'enabled' && rule.enabled) ||
                         (statusFilter === 'disabled' && !rule.enabled)
    return matchesSearch && matchesStatus
  })

  const RuleCard = ({ rule }: { rule: RoutingRule }) => (
    <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 shadow-sm hover:shadow-md transition-all duration-200">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className={`w-3 h-3 rounded-full ${rule.enabled ? 'bg-green-500' : 'bg-gray-400'}`}></div>
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-white text-lg">
              {rule.name}
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Priority {rule.priority}
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <span className={`px-3 py-1 text-xs rounded-full border font-medium ${getStatusColor(rule.enabled)}`}>
            {rule.enabled ? '✅ Enabled' : '⏸️ Disabled'}
          </span>
          <button
            onClick={() => toggleRule(rule.id)}
            className={`w-11 h-6 rounded-full relative transition-colors ${
              rule.enabled ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-600'
            }`}
          >
            <span
              className={`absolute w-5 h-5 bg-white rounded-full shadow transform transition-transform top-0.5 ${
                rule.enabled ? 'translate-x-5' : 'translate-x-0.5'
              }`}
            />
          </button>
        </div>
      </div>

      <p className="text-gray-600 dark:text-gray-300 text-sm mb-4">
        {rule.description}
      </p>

      {/* Conditions */}
      <div className="mb-4">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Conditions:</h4>
        <div className="flex flex-wrap gap-2">
          {rule.conditions.taskType?.map((type, index) => (
            <span key={index} className="px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 text-xs rounded-lg">
              Type: {type}
            </span>
          ))}
          {rule.conditions.keywords?.map((keyword, index) => (
            <span key={index} className="px-2 py-1 bg-purple-100 dark:bg-purple-900 text-purple-700 dark:text-purple-300 text-xs rounded-lg">
              "{keyword}"
            </span>
          ))}
          {rule.conditions.priority?.map((priority, index) => (
            <span key={index} className="px-2 py-1 bg-orange-100 dark:bg-orange-900 text-orange-700 dark:text-orange-300 text-xs rounded-lg">
              Priority: {priority}
            </span>
          ))}
          {rule.conditions.contentLength && (
            <span className="px-2 py-1 bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 text-xs rounded-lg">
              Length: {rule.conditions.contentLength.min ? `>${rule.conditions.contentLength.min}` : ''}
              {rule.conditions.contentLength.max ? `<${rule.conditions.contentLength.max}` : ''}
            </span>
          )}
        </div>
      </div>

      {/* Target */}
      <div className="mb-4">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Target:</h4>
        <div className="flex items-center space-x-2">
          <span className="px-3 py-1 bg-indigo-100 dark:bg-indigo-900 text-indigo-700 dark:text-indigo-300 text-sm rounded-lg font-medium">
            {rule.target.type === 'agent' ? '🤖' : '🧠'} {rule.target.value}
          </span>
          {rule.target.fallback && (
            <span className="text-xs text-gray-500 dark:text-gray-400">
              → Fallback: {rule.target.fallback}
            </span>
          )}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 pt-4 border-t border-gray-100 dark:border-gray-700">
        <div className="text-center">
          <div className="text-lg font-bold text-gray-900 dark:text-white">{rule.stats.totalRequests}</div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Requests</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-green-600">{rule.stats.successRate}%</div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Success</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-blue-600">{rule.stats.avgResponseTime}s</div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Avg Time</div>
        </div>
      </div>

      {rule.stats.lastUsed && (
        <div className="mt-3 text-xs text-gray-500 dark:text-gray-400 text-center">
          Last used: {formatTimeAgo(rule.stats.lastUsed)}
        </div>
      )}
    </div>
  )

  const ModelCard = ({ model }: { model: Model }) => (
    <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 shadow-sm hover:shadow-md transition-all duration-200">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center">
            <span className="text-white font-bold text-lg">
              {model.provider === 'OpenAI' ? '🔥' : model.provider === 'Anthropic' ? '🧠' : '💻'}
            </span>
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-white text-lg">
              {model.name}
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              {model.provider} • {model.version}
            </p>
          </div>
        </div>
        <span className={`px-3 py-1 text-xs rounded-full border font-medium ${getModelStatusColor(model.status)}`}>
          {model.status === 'available' ? '✅ Available' : 
           model.status === 'maintenance' ? '🔧 Maintenance' : '❌ Unavailable'}
        </span>
      </div>

      {/* Capabilities */}
      <div className="mb-4">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Capabilities:</h4>
        <div className="flex flex-wrap gap-2">
          {model.capabilities.map((capability, index) => (
            <span key={index} className="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-xs rounded-lg">
              {capability}
            </span>
          ))}
        </div>
      </div>

      {/* Pricing */}
      <div className="mb-4">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Pricing:</h4>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div>
            <span className="text-gray-500 dark:text-gray-400">Input:</span>
            <span className="ml-1 font-medium">{formatCurrency(model.pricing.inputTokens, model.pricing.currency)}/1K</span>
          </div>
          <div>
            <span className="text-gray-500 dark:text-gray-400">Output:</span>
            <span className="ml-1 font-medium">{formatCurrency(model.pricing.outputTokens, model.pricing.currency)}/1K</span>
          </div>
        </div>
      </div>

      {/* Limits */}
      <div className="mb-4">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Limits:</h4>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div>
            <span className="text-gray-500 dark:text-gray-400">Max Tokens:</span>
            <span className="ml-1 font-medium">{model.limits.maxTokens.toLocaleString()}</span>
          </div>
          <div>
            <span className="text-gray-500 dark:text-gray-400">RPM:</span>
            <span className="ml-1 font-medium">{model.limits.requestsPerMinute}</span>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 pt-4 border-t border-gray-100 dark:border-gray-700">
        <div className="text-center">
          <div className="text-lg font-bold text-gray-900 dark:text-white">{model.stats.totalRequests}</div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Requests</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-green-600">{model.stats.successRate}%</div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Success</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-blue-600">{model.stats.avgResponseTime}s</div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Avg Time</div>
        </div>
      </div>
    </div>
  )

  return (
    <div className="p-6 bg-gray-50 dark:bg-gray-900 min-h-screen">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
              Request Routing
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Configure intelligent task routing and model selection
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setIsCreatingRule(true)}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors flex items-center space-x-2"
            >
              <span>➕</span>
              <span>Add Rule</span>
            </button>
          </div>
        </div>

        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900 rounded-lg flex items-center justify-center">
                <span className="text-blue-600 dark:text-blue-400">🔄</span>
              </div>
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Active Rules</span>
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {routingRules.filter(r => r.enabled).length}
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 bg-green-100 dark:bg-green-900 rounded-lg flex items-center justify-center">
                <span className="text-green-600 dark:text-green-400">🧠</span>
              </div>
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Available Models</span>
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {models.filter(m => m.status === 'available').length}
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 bg-purple-100 dark:bg-purple-900 rounded-lg flex items-center justify-center">
                <span className="text-purple-600 dark:text-purple-400">📊</span>
              </div>
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Requests</span>
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {routingRules.reduce((acc, rule) => acc + rule.stats.totalRequests, 0).toLocaleString()}
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 bg-emerald-100 dark:bg-emerald-900 rounded-lg flex items-center justify-center">
                <span className="text-emerald-600 dark:text-emerald-400">✅</span>
              </div>
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Avg Success Rate</span>
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {routingRules.length > 0 ? 
                ((routingRules.reduce((acc, rule) => acc + rule.stats.successRate, 0) / routingRules.length).toFixed(1) + '%') 
                : '0%'
              }
            </p>
          </div>
        </div>

        {/* Tabs */}
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 mb-6">
          <div className="border-b border-gray-200 dark:border-gray-700">
            <nav className="flex space-x-8 px-6">
              {[
                { id: 'rules', label: 'Routing Rules', icon: '🔄' },
                { id: 'models', label: 'Models', icon: '🧠' },
                { id: 'analytics', label: 'Analytics', icon: '📊' }
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                      : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
                  }`}
                >
                  <span className="mr-2">{tab.icon}</span>
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>

          {/* Tab Content */}
          <div className="p-6">
            {activeTab === 'rules' && (
              <div>
                {/* Filters */}
                <div className="flex flex-col sm:flex-row gap-4 mb-6">
                  <div className="flex-1">
                    <input
                      type="text"
                      placeholder="Search routing rules..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:shadow-focus"
                    />
                  </div>
                  <select
                    value={statusFilter}
                    onChange={(e) => setStatusFilter(e.target.value as any)}
                    className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:shadow-focus"
                  >
                    <option value="all">All Rules</option>
                    <option value="enabled">Enabled Only</option>
                    <option value="disabled">Disabled Only</option>
                  </select>
                </div>

                {/* Rules Grid */}
                {filteredRules.length === 0 ? (
                  <div className="text-center py-12">
                    <div className="w-16 h-16 bg-gray-100 dark:bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-4">
                      <span className="text-gray-400 text-2xl">🔄</span>
                    </div>
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">No routing rules found</h3>
                    <p className="text-gray-500 dark:text-gray-400">Create your first routing rule to get started</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {filteredRules.map((rule) => (
                      <RuleCard key={rule.id} rule={rule} />
                    ))}
                  </div>
                )}
              </div>
            )}

            {activeTab === 'models' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {models.map((model) => (
                  <ModelCard key={model.id} model={model} />
                ))}
              </div>
            )}

            {activeTab === 'analytics' && (
              <div className="space-y-6">
                {/* Analytics placeholder */}
                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-12 text-center">
                  <div className="text-4xl mb-4">📊</div>
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                    Routing Analytics
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    Detailed analytics and performance metrics coming soon
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
