import { useState, useEffect } from 'react'

interface MetricData {
  timestamp: Date
  value: number
  metadata?: Record<string, any>
}

interface MetricDefinition {
  id: string
  name: string
  description: string
  unit: string
  type: 'gauge' | 'counter' | 'histogram' | 'summary'
  category: 'system' | 'agent' | 'task' | 'api' | 'business'
  data: MetricData[]
  currentValue: number
  trend: 'up' | 'down' | 'stable'
  trendPercentage: number
  thresholds?: {
    warning?: number
    critical?: number
  }
}

interface Dashboard {
  id: string
  name: string
  description: string
  widgets: DashboardWidget[]
  shared: boolean
  createdBy: string
  createdAt: Date
}

interface DashboardWidget {
  id: string
  type: 'chart' | 'number' | 'gauge' | 'table' | 'alert'
  title: string
  metricIds: string[]
  position: { x: number; y: number; w: number; h: number }
  config: Record<string, any>
}

const generateTimeSeriesData = (hours: number, baseValue: number, volatility: number = 0.1) => {
  const data: MetricData[] = []
  const now = new Date()
  
  for (let i = hours * 60; i >= 0; i--) {
    const timestamp = new Date(now.getTime() - i * 60000)
    const noise = (Math.random() - 0.5) * volatility * baseValue
    const trend = Math.sin(i * 0.01) * baseValue * 0.2
    const value = Math.max(0, baseValue + noise + trend)
    
    data.push({ timestamp, value })
  }
  
  return data
}

const mockMetrics: MetricDefinition[] = [
  {
    id: 'cpu_usage',
    name: 'CPU Usage',
    description: 'System CPU utilization percentage',
    unit: '%',
    type: 'gauge',
    category: 'system',
    data: generateTimeSeriesData(24, 45, 0.3),
    currentValue: 42.3,
    trend: 'down',
    trendPercentage: -5.2,
    thresholds: { warning: 70, critical: 90 }
  },
  {
    id: 'memory_usage',
    name: 'Memory Usage',
    description: 'System memory utilization',
    unit: 'GB',
    type: 'gauge',
    category: 'system',
    data: generateTimeSeriesData(24, 8.5, 0.1),
    currentValue: 8.2,
    trend: 'stable',
    trendPercentage: 0.8,
    thresholds: { warning: 12, critical: 15 }
  },
  {
    id: 'active_agents',
    name: 'Active Agents',
    description: 'Number of currently active AI agents',
    unit: 'count',
    type: 'gauge',
    category: 'agent',
    data: generateTimeSeriesData(24, 12, 0.2),
    currentValue: 11,
    trend: 'up',
    trendPercentage: 8.3,
    thresholds: { warning: 20, critical: 25 }
  },
  {
    id: 'task_completion_rate',
    name: 'Task Completion Rate',
    description: 'Percentage of successfully completed tasks',
    unit: '%',
    type: 'gauge',
    category: 'task',
    data: generateTimeSeriesData(24, 94.5, 0.05),
    currentValue: 95.2,
    trend: 'up',
    trendPercentage: 2.1,
    thresholds: { warning: 85, critical: 75 }
  },
  {
    id: 'avg_response_time',
    name: 'Average Response Time',
    description: 'Average API response time across all endpoints',
    unit: 'ms',
    type: 'gauge',
    category: 'api',
    data: generateTimeSeriesData(24, 145, 0.4),
    currentValue: 132,
    trend: 'down',
    trendPercentage: -12.5,
    thresholds: { warning: 300, critical: 500 }
  },
  {
    id: 'total_requests',
    name: 'Total Requests',
    description: 'Total number of API requests processed',
    unit: 'req/min',
    type: 'counter',
    category: 'api',
    data: generateTimeSeriesData(24, 245, 0.3),
    currentValue: 267,
    trend: 'up',
    trendPercentage: 15.7,
    thresholds: { warning: 500, critical: 750 }
  },
  {
    id: 'error_rate',
    name: 'Error Rate',
    description: 'Percentage of failed requests',
    unit: '%',
    type: 'gauge',
    category: 'api',
    data: generateTimeSeriesData(24, 2.1, 0.5),
    currentValue: 1.8,
    trend: 'down',
    trendPercentage: -14.3,
    thresholds: { warning: 5, critical: 10 }
  },
  {
    id: 'queue_length',
    name: 'Task Queue Length',
    description: 'Number of tasks waiting in queue',
    unit: 'tasks',
    type: 'gauge',
    category: 'task',
    data: generateTimeSeriesData(24, 15, 0.4),
    currentValue: 12,
    trend: 'down',
    trendPercentage: -20.0,
    thresholds: { warning: 50, critical: 100 }
  }
]

const mockDashboards: Dashboard[] = [
  {
    id: '1',
    name: 'System Overview',
    description: 'High-level system health and performance metrics',
    shared: true,
    createdBy: 'Admin',
    createdAt: new Date(Date.now() - 86400000),
    widgets: [
      {
        id: 'w1',
        type: 'number',
        title: 'CPU Usage',
        metricIds: ['cpu_usage'],
        position: { x: 0, y: 0, w: 3, h: 2 },
        config: { showTrend: true }
      },
      {
        id: 'w2',
        type: 'chart',
        title: 'Response Time Trend',
        metricIds: ['avg_response_time'],
        position: { x: 3, y: 0, w: 6, h: 4 },
        config: { chartType: 'line' }
      }
    ]
  },
  {
    id: '2',
    name: 'Agent Performance',
    description: 'Detailed metrics for AI agent performance',
    shared: false,
    createdBy: 'Current User',
    createdAt: new Date(Date.now() - 172800000),
    widgets: []
  }
]

export default function ModernMetricsPage() {
  const [metrics, setMetrics] = useState<MetricDefinition[]>(mockMetrics)
  const [dashboards, setDashboards] = useState<Dashboard[]>(mockDashboards)
  const [activeTab, setActiveTab] = useState<'overview' | 'explorer' | 'dashboards' | 'alerts'>('overview')
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['cpu_usage', 'memory_usage', 'avg_response_time'])
  const [timeRange, setTimeRange] = useState<'1h' | '6h' | '24h' | '7d' | '30d'>('24h')
  const [categoryFilter, setCategoryFilter] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [isExporting, setIsExporting] = useState(false)
  const [customDateRange, setCustomDateRange] = useState({ start: '', end: '' })
  const [refreshInterval, setRefreshInterval] = useState(30)
  const [autoRefresh, setAutoRefresh] = useState(true)

  // Simulate real-time metric updates
  useEffect(() => {
    if (!autoRefresh) return

    const interval = setInterval(() => {
      setMetrics(prev => prev.map(metric => {
        const lastValue = metric.data[metric.data.length - 1]?.value || metric.currentValue
        const noise = (Math.random() - 0.5) * 0.1 * lastValue
        const newValue = Math.max(0, lastValue + noise)
        
        return {
          ...metric,
          currentValue: newValue,
          data: [
            ...metric.data.slice(-100),
            { timestamp: new Date(), value: newValue }
          ]
        }
      }))
    }, refreshInterval * 1000)

    return () => clearInterval(interval)
  }, [autoRefresh, refreshInterval])

  const getMetricTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return '📈'
      case 'down': return '📉'
      case 'stable': return '➡️'
      default: return '➡️'
    }
  }

  const getMetricTrendColor = (trend: string, isGoodTrend: boolean = true) => {
    if (trend === 'stable') return 'text-gray-600 dark:text-gray-400'
    
    const isPositiveTrend = trend === 'up'
    if ((isPositiveTrend && isGoodTrend) || (!isPositiveTrend && !isGoodTrend)) {
      return 'text-green-600 dark:text-green-400'
    } else {
      return 'text-red-600 dark:text-red-400'
    }
  }

  const getThresholdStatus = (value: number, thresholds?: { warning?: number; critical?: number }) => {
    if (!thresholds) return 'normal'
    if (thresholds.critical && value >= thresholds.critical) return 'critical'
    if (thresholds.warning && value >= thresholds.warning) return 'warning'
    return 'normal'
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'critical': return 'bg-red-100 text-red-800 border-red-200'
      case 'warning': return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      case 'normal': return 'bg-green-100 text-green-800 border-green-200'
      default: return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  const formatValue = (value: number, unit: string) => {
    if (unit === '%' || unit === 'ms' || unit === 'count' || unit === 'tasks' || unit === 'req/min') {
      return `${value.toFixed(1)} ${unit}`
    }
    if (unit === 'GB') {
      return `${value.toFixed(2)} ${unit}`
    }
    return `${value.toFixed(0)} ${unit}`
  }

  const exportMetrics = async () => {
    setIsExporting(true)
    
    try {
      // Simulate export process
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      const exportData = {
        timestamp: new Date().toISOString(),
        timeRange,
        metrics: selectedMetrics.map(id => {
          const metric = metrics.find(m => m.id === id)
          return {
            id: metric?.id,
            name: metric?.name,
            currentValue: metric?.currentValue,
            data: metric?.data.slice(-100) // Last 100 points
          }
        })
      }
      
      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `metrics-export-${new Date().toISOString().split('T')[0]}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      
    } catch (error) {
      console.error('Export failed:', error)
    } finally {
      setIsExporting(false)
    }
  }

  const filteredMetrics = metrics.filter(metric => {
    const matchesCategory = categoryFilter === 'all' || metric.category === categoryFilter
    const matchesSearch = searchQuery === '' || 
      metric.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      metric.description.toLowerCase().includes(searchQuery.toLowerCase())
    
    return matchesCategory && matchesSearch
  })

  const SimpleChart = ({ 
    data, 
    color, 
    height = 80,
    showGrid = false 
  }: {
    data: MetricData[]
    color: string
    height?: number
    showGrid?: boolean
  }) => {
    if (data.length === 0) return null

    const values = data.map(d => d.value)
    const max = Math.max(...values)
    const min = Math.min(...values)
    const range = max - min || 1

    return (
      <div className="relative" style={{ height }}>
        <svg width="100%" height="100%" className="absolute inset-0">
          {showGrid && (
            <defs>
              <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#e5e7eb" strokeWidth="0.5"/>
              </pattern>
            </defs>
          )}
          {showGrid && <rect width="100%" height="100%" fill="url(#grid)" />}
          
          <polyline
            fill="none"
            stroke={color}
            strokeWidth="2"
            points={values
              .map((value, index) => {
                const x = (index / (values.length - 1)) * 100
                const y = 100 - ((value - min) / range) * 90 - 5
                return `${x},${y}`
              })
              .join(' ')}
          />
          
          {/* Data points */}
          {values.map((value, index) => {
            const x = (index / (values.length - 1)) * 100
            const y = 100 - ((value - min) / range) * 90 - 5
            return (
              <circle
                key={index}
                cx={`${x}%`}
                cy={y}
                r="2"
                fill={color}
                opacity="0.6"
              />
            )
          })}
        </svg>
      </div>
    )
  }

  const MetricCard = ({ metric }: { metric: MetricDefinition }) => {
    const status = getThresholdStatus(metric.currentValue, metric.thresholds)
    const isGoodTrend = ['task_completion_rate', 'active_agents', 'total_requests'].includes(metric.id)
    
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 shadow-sm hover:shadow-md transition-all duration-200">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-white text-lg">
              {metric.name}
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              {metric.description}
            </p>
          </div>
          <span className={`px-2 py-1 text-xs rounded-full border font-medium ${getStatusColor(status)}`}>
            {status}
          </span>
        </div>

        {/* Current Value */}
        <div className="flex items-center justify-between mb-4">
          <div className="text-3xl font-bold text-gray-900 dark:text-white">
            {formatValue(metric.currentValue, metric.unit)}
          </div>
          <div className={`flex items-center space-x-1 text-sm ${getMetricTrendColor(metric.trend, isGoodTrend)}`}>
            <span>{getMetricTrendIcon(metric.trend)}</span>
            <span>{Math.abs(metric.trendPercentage).toFixed(1)}%</span>
          </div>
        </div>

        {/* Mini Chart */}
        <div className="mb-4">
          <SimpleChart 
            data={metric.data.slice(-20)} 
            color="#3b82f6"
            height={60}
          />
        </div>

        {/* Thresholds */}
        {metric.thresholds && (
          <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
            {metric.thresholds.warning && (
              <div className="flex justify-between">
                <span>Warning:</span>
                <span>{formatValue(metric.thresholds.warning, metric.unit)}</span>
              </div>
            )}
            {metric.thresholds.critical && (
              <div className="flex justify-between">
                <span>Critical:</span>
                <span>{formatValue(metric.thresholds.critical, metric.unit)}</span>
              </div>
            )}
          </div>
        )}
      </div>
    )
  }

  const categories = ['all', ...Array.from(new Set(metrics.map(m => m.category)))]

  return (
    <div className="p-6 bg-gray-50 dark:bg-gray-900 min-h-screen">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
              Performance Metrics
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Comprehensive system and application performance analytics
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <label className="text-sm text-gray-600 dark:text-gray-400">Auto-refresh:</label>
              <button
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`w-11 h-6 rounded-full relative transition-colors ${
                  autoRefresh ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-600'
                }`}
              >
                <span
                  className={`absolute w-5 h-5 bg-white rounded-full shadow transform transition-transform top-0.5 ${
                    autoRefresh ? 'translate-x-5' : 'translate-x-0.5'
                  }`}
                />
              </button>
            </div>
            
            <select
              value={refreshInterval}
              onChange={(e) => setRefreshInterval(Number(e.target.value))}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white text-sm focus:outline-none focus:shadow-focus"
            >
              <option value={10}>10s</option>
              <option value={30}>30s</option>
              <option value={60}>1m</option>
              <option value={300}>5m</option>
            </select>

            <button
              onClick={exportMetrics}
              disabled={isExporting}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 transition-colors flex items-center space-x-2"
            >
              {isExporting ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>Exporting...</span>
                </>
              ) : (
                <>
                  <span>📊</span>
                  <span>Export</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900 rounded-lg flex items-center justify-center">
                <span className="text-blue-600 dark:text-blue-400">📊</span>
              </div>
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Metrics</span>
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">{metrics.length}</p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 bg-green-100 dark:bg-green-900 rounded-lg flex items-center justify-center">
                <span className="text-green-600 dark:text-green-400">✅</span>
              </div>
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Healthy</span>
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {metrics.filter(m => getThresholdStatus(m.currentValue, m.thresholds) === 'normal').length}
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 bg-yellow-100 dark:bg-yellow-900 rounded-lg flex items-center justify-center">
                <span className="text-yellow-600 dark:text-yellow-400">⚠️</span>
              </div>
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Warnings</span>
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {metrics.filter(m => getThresholdStatus(m.currentValue, m.thresholds) === 'warning').length}
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 bg-red-100 dark:bg-red-900 rounded-lg flex items-center justify-center">
                <span className="text-red-600 dark:text-red-400">🚨</span>
              </div>
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Critical</span>
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {metrics.filter(m => getThresholdStatus(m.currentValue, m.thresholds) === 'critical').length}
            </p>
          </div>
        </div>

        {/* Tabs */}
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 mb-6">
          <div className="border-b border-gray-200 dark:border-gray-700">
            <nav className="flex space-x-8 px-6">
              {[
                { id: 'overview', label: 'Overview', icon: '📋' },
                { id: 'explorer', label: 'Metric Explorer', icon: '🔍' },
                { id: 'dashboards', label: 'Dashboards', icon: '📊' },
                { id: 'alerts', label: 'Alert Rules', icon: '🚨' }
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
            {activeTab === 'overview' && (
              <div>
                {/* Key Metrics Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {metrics.slice(0, 6).map((metric) => (
                    <MetricCard key={metric.id} metric={metric} />
                  ))}
                </div>
              </div>
            )}

            {activeTab === 'explorer' && (
              <div>
                {/* Filters */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <input
                    type="text"
                    placeholder="Search metrics..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:shadow-focus"
                  />
                  
                  <select
                    value={categoryFilter}
                    onChange={(e) => setCategoryFilter(e.target.value)}
                    className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:shadow-focus"
                  >
                    {categories.map(category => (
                      <option key={category} value={category}>
                        {category === 'all' ? 'All Categories' : category.charAt(0).toUpperCase() + category.slice(1)}
                      </option>
                    ))}
                  </select>

                  <select
                    value={timeRange}
                    onChange={(e) => setTimeRange(e.target.value as any)}
                    className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:shadow-focus"
                  >
                    <option value="1h">Last Hour</option>
                    <option value="6h">Last 6 Hours</option>
                    <option value="24h">Last 24 Hours</option>
                    <option value="7d">Last 7 Days</option>
                    <option value="30d">Last 30 Days</option>
                  </select>
                </div>

                {/* Metrics Grid */}
                {filteredMetrics.length === 0 ? (
                  <div className="text-center py-12">
                    <div className="w-16 h-16 bg-gray-100 dark:bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-4">
                      <span className="text-gray-400 text-2xl">📊</span>
                    </div>
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">No metrics found</h3>
                    <p className="text-gray-500 dark:text-gray-400">Try adjusting your search criteria</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {filteredMetrics.map((metric) => (
                      <MetricCard key={metric.id} metric={metric} />
                    ))}
                  </div>
                )}
              </div>
            )}

            {activeTab === 'dashboards' && (
              <div>
                {/* Dashboard List */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {dashboards.map((dashboard) => (
                    <div key={dashboard.id} className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
                      <div className="flex items-start justify-between mb-4">
                        <div>
                          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                            {dashboard.name}
                          </h3>
                          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                            {dashboard.description}
                          </p>
                        </div>
                        <span className={`px-2 py-1 text-xs rounded-full ${
                          dashboard.shared 
                            ? 'bg-green-100 text-green-800 border border-green-200' 
                            : 'bg-gray-100 text-gray-800 border border-gray-200'
                        }`}>
                          {dashboard.shared ? '🌐 Shared' : '🔒 Private'}
                        </span>
                      </div>
                      
                      <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400 mb-4">
                        <span>Created by {dashboard.createdBy}</span>
                        <span>{dashboard.widgets.length} widgets</span>
                      </div>
                      
                      <div className="flex space-x-2">
                        <button className="flex-1 px-3 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors text-sm">
                          Open Dashboard
                        </button>
                        <button className="px-3 py-2 border border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-400 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors text-sm">
                          Edit
                        </button>
                      </div>
                    </div>
                  ))}
                  
                  {/* Create Dashboard Card */}
                  <div className="bg-gradient-to-r from-gray-50 to-blue-50 dark:from-gray-800 dark:to-blue-900/20 rounded-xl p-6 border-2 border-dashed border-gray-300 dark:border-gray-600 flex items-center justify-center">
                    <div className="text-center">
                      <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center mx-auto mb-3">
                        <span className="text-blue-600 dark:text-blue-400 text-xl">➕</span>
                      </div>
                      <h3 className="font-medium text-gray-900 dark:text-white mb-2">Create Dashboard</h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                        Build custom dashboards with your metrics
                      </p>
                      <button className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors text-sm">
                        New Dashboard
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'alerts' && (
              <div className="text-center py-12">
                <div className="w-16 h-16 bg-orange-100 dark:bg-orange-900 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-orange-600 dark:text-orange-400 text-2xl">🚨</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                  Alert Rules
                </h3>
                <p className="text-gray-600 dark:text-gray-400 mb-6">
                  Configure automated alerts based on metric thresholds
                </p>
                <button className="px-6 py-3 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors">
                  Configure Alerts
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
