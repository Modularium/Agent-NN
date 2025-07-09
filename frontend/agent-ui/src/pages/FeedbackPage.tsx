import { useState, useEffect } from 'react'

interface FeedbackEntry {
  id: string
  type: 'bug' | 'feature' | 'improvement' | 'general'
  title: string
  description: string
  rating?: number
  category: string
  priority: 'low' | 'medium' | 'high'
  status: 'open' | 'in-progress' | 'resolved' | 'closed'
  submittedAt: Date
  submittedBy: {
    name: string
    email: string
    role: string
  }
  sessionId?: string
  agent?: string
  attachments?: Array<{
    name: string
    size: number
    type: string
  }>
  responses?: Array<{
    id: string
    message: string
    author: string
    timestamp: Date
    isAdmin: boolean
  }>
  upvotes: number
  tags: string[]
}

interface FeedbackStats {
  total: number
  byType: { [key: string]: number }
  byStatus: { [key: string]: number }
  avgRating: number
  recentCount: number
}

const mockFeedback: FeedbackEntry[] = [
  {
    id: '1',
    type: 'bug',
    title: 'Docker container deployment fails with timeout',
    description: 'When deploying containers through the DockerMaster agent, I frequently encounter timeout errors after 30 seconds. This happens especially with larger images.',
    rating: 2,
    category: 'Agent Performance',
    priority: 'high',
    status: 'in-progress',
    submittedAt: new Date(Date.now() - 7200000),
    submittedBy: {
      name: 'John Doe',
      email: 'john@example.com',
      role: 'Developer'
    },
    sessionId: 'sess_123',
    agent: 'DockerMaster',
    upvotes: 5,
    tags: ['docker', 'timeout', 'deployment'],
    responses: [
      {
        id: 'r1',
        message: 'Thank you for reporting this. We\'re investigating the timeout issues with large container images.',
        author: 'Support Team',
        timestamp: new Date(Date.now() - 3600000),
        isAdmin: true
      }
    ]
  },
  {
    id: '2',
    type: 'feature',
    title: 'Add bulk task management',
    description: 'It would be great to have the ability to manage multiple tasks at once - start, stop, or delete multiple tasks with a single action.',
    rating: 5,
    category: 'User Interface',
    priority: 'medium',
    status: 'open',
    submittedAt: new Date(Date.now() - 14400000),
    submittedBy: {
      name: 'Sarah Wilson',
      email: 'sarah@example.com',
      role: 'Admin'
    },
    upvotes: 12,
    tags: ['ui', 'productivity', 'bulk-actions']
  },
  {
    id: '3',
    type: 'improvement',
    title: 'Faster response times for CodeAnalyzer',
    description: 'The CodeAnalyzer agent takes quite long to analyze large codebases. Could we optimize this or add progress indicators?',
    rating: 4,
    category: 'Performance',
    priority: 'medium',
    status: 'resolved',
    submittedAt: new Date(Date.now() - 86400000),
    submittedBy: {
      name: 'Mike Chen',
      email: 'mike@example.com',
      role: 'Developer'
    },
    agent: 'CodeAnalyzer',
    upvotes: 8,
    tags: ['performance', 'code-analysis', 'progress']
  },
  {
    id: '4',
    type: 'general',
    title: 'Love the new chat interface!',
    description: 'The updated chat interface is really intuitive and makes working with agents much easier. Great job!',
    rating: 5,
    category: 'User Experience',
    priority: 'low',
    status: 'closed',
    submittedAt: new Date(Date.now() - 172800000),
    submittedBy: {
      name: 'Lisa Zhang',
      email: 'lisa@example.com',
      role: 'Designer'
    },
    upvotes: 15,
    tags: ['ui', 'chat', 'positive']
  }
]

export default function ModernFeedbackPage() {
  const [feedback, setFeedback] = useState<FeedbackEntry[]>(mockFeedback)
  const [filteredFeedback, setFilteredFeedback] = useState<FeedbackEntry[]>(mockFeedback)
  const [activeTab, setActiveTab] = useState<'list' | 'submit' | 'analytics'>('list')
  const [typeFilter, setTypeFilter] = useState<string>('all')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [sortBy, setSortBy] = useState<'newest' | 'oldest' | 'rating' | 'upvotes'>('newest')
  // const [selectedFeedback, setSelectedFeedback] = useState<string | null>(null)

  // New feedback form state
  const [newFeedback, setNewFeedback] = useState({
    type: 'general' as FeedbackEntry['type'],
    title: '',
    description: '',
    category: '',
    rating: 5,
    tags: [] as string[]
  })
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitSuccess, setSubmitSuccess] = useState(false)

  // Filter and sort feedback
  useEffect(() => {
    let filtered = feedback.filter(item => {
      const matchesType = typeFilter === 'all' || item.type === typeFilter
      const matchesStatus = statusFilter === 'all' || item.status === statusFilter
      const matchesSearch = searchQuery === '' || 
        item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
      
      return matchesType && matchesStatus && matchesSearch
    })

    // Sort feedback
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'oldest':
          return a.submittedAt.getTime() - b.submittedAt.getTime()
        case 'rating':
          return (b.rating || 0) - (a.rating || 0)
        case 'upvotes':
          return b.upvotes - a.upvotes
        default:
          return b.submittedAt.getTime() - a.submittedAt.getTime()
      }
    })

    setFilteredFeedback(filtered)
  }, [feedback, typeFilter, statusFilter, searchQuery, sortBy])

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'bug': return 'bg-red-100 text-red-800 border-red-200'
      case 'feature': return 'bg-blue-100 text-blue-800 border-blue-200'
      case 'improvement': return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      case 'general': return 'bg-green-100 text-green-800 border-green-200'
      default: return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'open': return 'bg-blue-100 text-blue-800 border-blue-200'
      case 'in-progress': return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      case 'resolved': return 'bg-green-100 text-green-800 border-green-200'
      case 'closed': return 'bg-gray-100 text-gray-800 border-gray-200'
      default: return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'bg-red-500'
      case 'medium': return 'bg-yellow-500'
      case 'low': return 'bg-green-500'
      default: return 'bg-gray-500'
    }
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'bug': return '🐛'
      case 'feature': return '✨'
      case 'improvement': return '⚡'
      case 'general': return '💬'
      default: return '📝'
    }
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

  const renderStars = (rating: number) => {
    return Array.from({ length: 5 }, (_, i) => (
      <span key={i} className={`text-lg ${i < rating ? 'text-yellow-400' : 'text-gray-300'}`}>
        ⭐
      </span>
    ))
  }

  const handleSubmitFeedback = async () => {
    if (!newFeedback.title.trim() || !newFeedback.description.trim()) return

    setIsSubmitting(true)
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      const feedbackEntry: FeedbackEntry = {
        id: Date.now().toString(),
        ...newFeedback,
        priority: 'medium',
        status: 'open',
        submittedAt: new Date(),
        submittedBy: {
          name: 'Current User',
          email: 'user@example.com',
          role: 'User'
        },
        upvotes: 0
      }
      
      setFeedback(prev => [feedbackEntry, ...prev])
      setNewFeedback({
        type: 'general',
        title: '',
        description: '',
        category: '',
        rating: 5,
        tags: []
      })
      setSubmitSuccess(true)
      setTimeout(() => setSubmitSuccess(false), 3000)
      
    } catch (error) {
      console.error('Failed to submit feedback:', error)
    } finally {
      setIsSubmitting(false)
    }
  }

  const upvoteFeedback = (id: string) => {
    setFeedback(prev => prev.map(item => 
      item.id === id ? { ...item, upvotes: item.upvotes + 1 } : item
    ))
  }

  const stats: FeedbackStats = {
    total: feedback.length,
    byType: {
      bug: feedback.filter(f => f.type === 'bug').length,
      feature: feedback.filter(f => f.type === 'feature').length,
      improvement: feedback.filter(f => f.type === 'improvement').length,
      general: feedback.filter(f => f.type === 'general').length
    },
    byStatus: {
      open: feedback.filter(f => f.status === 'open').length,
      'in-progress': feedback.filter(f => f.status === 'in-progress').length,
      resolved: feedback.filter(f => f.status === 'resolved').length,
      closed: feedback.filter(f => f.status === 'closed').length
    },
    avgRating: feedback.reduce((acc, f) => acc + (f.rating || 0), 0) / feedback.filter(f => f.rating).length,
    recentCount: feedback.filter(f => f.submittedAt.getTime() > Date.now() - 24 * 60 * 60 * 1000).length
  }

  const FeedbackCard = ({ item }: { item: FeedbackEntry }) => (
    <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 shadow-sm hover:shadow-md transition-all duration-200">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-start space-x-3">
          <div className={`w-1 h-16 rounded-full ${getPriorityColor(item.priority)}`}></div>
          <div className="flex-1">
            <div className="flex items-center space-x-2 mb-2">
              <span className={`px-2 py-1 text-xs rounded-full border font-medium ${getTypeColor(item.type)}`}>
                {getTypeIcon(item.type)} {item.type}
              </span>
              <span className={`px-2 py-1 text-xs rounded-full border font-medium ${getStatusColor(item.status)}`}>
                {item.status}
              </span>
              {item.rating && (
                <div className="flex items-center space-x-1">
                  {renderStars(item.rating)}
                </div>
              )}
            </div>
            <h3 className="font-semibold text-gray-900 dark:text-white text-lg mb-2">
              {item.title}
            </h3>
            <p className="text-gray-600 dark:text-gray-300 text-sm line-clamp-3">
              {item.description}
            </p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={() => upvoteFeedback(item.id)}
            className="flex items-center space-x-1 px-2 py-1 text-sm text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-lg transition-colors"
          >
            <span>👍</span>
            <span>{item.upvotes}</span>
          </button>
        </div>
      </div>

      {/* Tags */}
      {item.tags.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-4">
          {item.tags.map((tag, index) => (
            <span
              key={index}
              className="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-xs rounded-lg"
            >
              #{tag}
            </span>
          ))}
        </div>
      )}

      {/* Meta information */}
      <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400 pt-4 border-t border-gray-100 dark:border-gray-700">
        <div className="flex items-center space-x-4">
          <span>By {item.submittedBy.name}</span>
          <span>•</span>
          <span>{item.category}</span>
          {item.agent && (
            <>
              <span>•</span>
              <span className="text-blue-600 dark:text-blue-400">{item.agent}</span>
            </>
          )}
        </div>
        <span>{formatTimeAgo(item.submittedAt)}</span>
      </div>

      {/* Responses */}
      {item.responses && item.responses.length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-700">
          <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Latest Response:
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
            <p className="text-sm text-gray-600 dark:text-gray-300">
              {item.responses[item.responses.length - 1].message}
            </p>
            <div className="flex items-center justify-between mt-2 text-xs text-gray-500 dark:text-gray-400">
              <span>{item.responses[item.responses.length - 1].author}</span>
              <span>{formatTimeAgo(item.responses[item.responses.length - 1].timestamp)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )

  return (
    <div className="p-6 bg-gray-50 dark:bg-gray-900 min-h-screen">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
              User Feedback
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Collect, manage, and respond to user feedback
            </p>
          </div>
        </div>

        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900 rounded-lg flex items-center justify-center">
                <span className="text-blue-600 dark:text-blue-400">📝</span>
              </div>
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Feedback</span>
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">{stats.total}</p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 bg-yellow-100 dark:bg-yellow-900 rounded-lg flex items-center justify-center">
                <span className="text-yellow-600 dark:text-yellow-400">⭐</span>
              </div>
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Avg Rating</span>
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {stats.avgRating ? stats.avgRating.toFixed(1) : 'N/A'}
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 bg-green-100 dark:bg-green-900 rounded-lg flex items-center justify-center">
                <span className="text-green-600 dark:text-green-400">✅</span>
              </div>
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Resolved</span>
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">{stats.byStatus.resolved}</p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 bg-purple-100 dark:bg-purple-900 rounded-lg flex items-center justify-center">
                <span className="text-purple-600 dark:text-purple-400">🕒</span>
              </div>
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Recent (24h)</span>
            </div>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">{stats.recentCount}</p>
          </div>
        </div>

        {/* Tabs */}
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 mb-6">
          <div className="border-b border-gray-200 dark:border-gray-700">
            <nav className="flex space-x-8 px-6">
              {[
                { id: 'list', label: 'Feedback List', icon: '📋' },
                { id: 'submit', label: 'Submit Feedback', icon: '✍️' },
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
            {activeTab === 'list' && (
              <div>
                {/* Filters */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                  <input
                    type="text"
                    placeholder="Search feedback..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:shadow-focus"
                  />
                  
                  <select
                    value={typeFilter}
                    onChange={(e) => setTypeFilter(e.target.value)}
                    className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:shadow-focus"
                  >
                    <option value="all">All Types</option>
                    <option value="bug">🐛 Bug Reports</option>
                    <option value="feature">✨ Feature Requests</option>
                    <option value="improvement">⚡ Improvements</option>
                    <option value="general">💬 General</option>
                  </select>

                  <select
                    value={statusFilter}
                    onChange={(e) => setStatusFilter(e.target.value)}
                    className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:shadow-focus"
                  >
                    <option value="all">All Status</option>
                    <option value="open">Open</option>
                    <option value="in-progress">In Progress</option>
                    <option value="resolved">Resolved</option>
                    <option value="closed">Closed</option>
                  </select>

                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value as any)}
                    className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:shadow-focus"
                  >
                    <option value="newest">Newest First</option>
                    <option value="oldest">Oldest First</option>
                    <option value="rating">Highest Rated</option>
                    <option value="upvotes">Most Upvotes</option>
                  </select>
                </div>

                {/* Feedback List */}
                {filteredFeedback.length === 0 ? (
                  <div className="text-center py-12">
                    <div className="w-16 h-16 bg-gray-100 dark:bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-4">
                      <span className="text-gray-400 text-2xl">📝</span>
                    </div>
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">No feedback found</h3>
                    <p className="text-gray-500 dark:text-gray-400">Try adjusting your search criteria</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {filteredFeedback.map((item) => (
                      <FeedbackCard key={item.id} item={item} />
                    ))}
                  </div>
                )}
              </div>
            )}

            {activeTab === 'submit' && (
              <div className="max-w-2xl mx-auto">
                {submitSuccess && (
                  <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4 mb-6">
                    <div className="flex items-center">
                      <span className="text-green-500 mr-3">✅</span>
                      <div>
                        <h3 className="text-sm font-medium text-green-800 dark:text-green-200">
                          Feedback Submitted Successfully
                        </h3>
                        <p className="text-sm text-green-700 dark:text-green-300 mt-1">
                          Thank you for your feedback! We'll review it and get back to you soon.
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                <div className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Feedback Type *
                    </label>
                    <select
                      value={newFeedback.type}
                      onChange={(e) => setNewFeedback(prev => ({ ...prev, type: e.target.value as any }))}
                      className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:shadow-focus"
                    >
                      <option value="general">💬 General Feedback</option>
                      <option value="bug">🐛 Bug Report</option>
                      <option value="feature">✨ Feature Request</option>
                      <option value="improvement">⚡ Improvement Suggestion</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Title *
                    </label>
                    <input
                      type="text"
                      value={newFeedback.title}
                      onChange={(e) => setNewFeedback(prev => ({ ...prev, title: e.target.value }))}
                      placeholder="Brief description of your feedback"
                      className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:shadow-focus"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Description *
                    </label>
                    <textarea
                      value={newFeedback.description}
                      onChange={(e) => setNewFeedback(prev => ({ ...prev, description: e.target.value }))}
                      placeholder="Detailed description of your feedback..."
                      rows={6}
                      className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:shadow-focus resize-none"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Category
                    </label>
                    <input
                      type="text"
                      value={newFeedback.category}
                      onChange={(e) => setNewFeedback(prev => ({ ...prev, category: e.target.value }))}
                      placeholder="e.g., User Interface, Performance, Documentation"
                      className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:shadow-focus"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Rating
                    </label>
                    <div className="flex items-center space-x-2">
                      {Array.from({ length: 5 }, (_, i) => (
                        <button
                          key={i}
                          onClick={() => setNewFeedback(prev => ({ ...prev, rating: i + 1 }))}
                          className={`text-2xl ${i < newFeedback.rating ? 'text-yellow-400' : 'text-gray-300'} hover:text-yellow-400 transition-colors`}
                        >
                          ⭐
                        </button>
                      ))}
                      <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">
                        {newFeedback.rating} out of 5
                      </span>
                    </div>
                  </div>

                  <div className="flex gap-4">
                    <button
                      onClick={() => setNewFeedback({
                        type: 'general',
                        title: '',
                        description: '',
                        category: '',
                        rating: 5,
                        tags: []
                      })}
                      className="px-6 py-3 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                    >
                      Reset
                    </button>
                    
                    <button
                      onClick={handleSubmitFeedback}
                      disabled={!newFeedback.title.trim() || !newFeedback.description.trim() || isSubmitting}
                      className="flex-1 px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center space-x-2"
                    >
                      {isSubmitting ? (
                        <>
                          <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                          <span>Submitting...</span>
                        </>
                      ) : (
                        <>
                          <span>✍️</span>
                          <span>Submit Feedback</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'analytics' && (
              <div className="space-y-6">
                {/* Type Distribution */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                      Feedback by Type
                    </h3>
                    <div className="space-y-3">
                      {Object.entries(stats.byType).map(([type, count]) => (
                        <div key={type} className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            <span>{getTypeIcon(type)}</span>
                            <span className="capitalize">{type}</span>
                          </div>
                          <span className="font-medium">{count}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                      Status Distribution
                    </h3>
                    <div className="space-y-3">
                      {Object.entries(stats.byStatus).map(([status, count]) => (
                        <div key={status} className="flex items-center justify-between">
                          <span className="capitalize">{status.replace('-', ' ')}</span>
                          <span className="font-medium">{count}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Recent Trends */}
                <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-12 text-center">
                  <div className="text-4xl mb-4">📈</div>
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                    Detailed Analytics
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    Advanced analytics and trends coming soon
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
