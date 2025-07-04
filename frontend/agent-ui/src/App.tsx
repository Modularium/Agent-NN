import { useState } from 'react'

// Navigation Items
const navigationItems = [
  { path: '/', name: 'Chat', icon: '💬', end: true },
  { path: '/agents', name: 'Agents', icon: '🤖' },
  { path: '/tasks', name: 'Tasks', icon: '📋' },
  { path: '/routing', name: 'Routing', icon: '🔄' },
  { path: '/feedback', name: 'Feedback', icon: '💭' },
  { path: '/monitoring', name: 'Monitoring', icon: '📊' },
  { path: '/metrics', name: 'Metrics', icon: '📈' },
  { path: '/settings', name: 'Settings', icon: '⚙️' },
  { path: '/admin', name: 'Admin', icon: '👤' },
  { path: '/debug', name: 'Debug', icon: '🔧' },
]

// Mock Data
const mockDashboardData = [
  { label: 'Active Agents', value: '12', change: '+2 from yesterday', icon: '🤖', color: 'blue' },
  { label: 'Completed Tasks', value: '847', change: '94% success rate', icon: '✅', color: 'green' },
  { label: 'Response Time', value: '0.3s', change: 'Average latency', icon: '⚡', color: 'purple' },
  { label: 'Uptime', value: '99.8%', change: 'Last 30 days', icon: '🔺', color: 'emerald' },
]

const mockRecentActivity = [
  { id: '1', action: 'Agent DockerMaster completed task #1247', time: '2 minutes ago', type: 'success' },
  { id: '2', action: 'New agent CodeAnalyzer v1.8.2 deployed', time: '15 minutes ago', type: 'info' },
  { id: '3', action: 'Task routing updated for container operations', time: '1 hour ago', type: 'update' },
  { id: '4', action: 'System performance metrics exported', time: '2 hours ago', type: 'info' },
]

// Sidebar Component
function ModernSidebar({ open, setOpen, activeItem, setActiveItem }: any) {
  const linkClass = (isActive: boolean) =>
    `flex items-center gap-3 px-4 py-3 mx-2 rounded-xl transition-all duration-200 hover:bg-gradient-to-r hover:from-blue-50 hover:to-indigo-50 hover:shadow-sm group cursor-pointer ${
      isActive 
        ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-lg shadow-blue-500/25' 
        : 'text-gray-700 hover:text-blue-600'
    }`

  return (
    <>
      {/* Overlay for mobile */}
      {open && (
        <div 
          className="fixed inset-0 bg-black/50 z-30 sm:hidden" 
          onClick={() => setOpen(false)}
        />
      )}
      
      <nav
        className={`${
          open ? 'translate-x-0' : '-translate-x-full'
        } sm:translate-x-0 w-64 bg-white border-r border-gray-200 h-full fixed sm:static z-40 transition-transform duration-300 ease-in-out shadow-xl sm:shadow-none`}
        aria-label="Main navigation"
      >
        <div className="p-6">
          {/* Logo/Header */}
          <div className="flex items-center gap-3 mb-8">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
              <span className="text-white font-bold text-lg">A</span>
            </div>
            <div>
              <h1 className="font-bold text-xl text-gray-900">Agent-NN</h1>
              <p className="text-sm text-gray-500">AI Management</p>
            </div>
          </div>

          {/* Close button for mobile */}
          <button
            className="sm:hidden absolute top-4 right-4 p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg"
            onClick={() => setOpen(false)}
            aria-label="Close menu"
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Navigation */}
        <div className="px-4 pb-6">
          <ul className="space-y-2">
            {navigationItems.map((item) => (
              <li key={item.path}>
                <div
                  className={linkClass(activeItem === item.path)} 
                  onClick={() => {
                    setActiveItem(item.path)
                    setOpen(false)
                  }}
                >
                  <span className="text-xl">{item.icon}</span>
                  <span className="font-medium">{item.name}</span>
                  {activeItem === item.path && (
                    <div className="ml-auto w-2 h-2 bg-white rounded-full"></div>
                  )}
                </div>
              </li>
            ))}
          </ul>
        </div>

        {/* Bottom section */}
        <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-r from-gray-50 to-blue-50 border-t border-gray-200">
          <div className="text-center">
            <p className="text-xs text-gray-500 mb-2">System Status</p>
            <div className="flex items-center justify-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-gray-600 font-medium">Online</span>
            </div>
          </div>
        </div>
      </nav>
    </>
  )
}

// Dashboard Component
function Dashboard() {
  return (
    <div className="p-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Dashboard</h1>
          <p className="text-gray-600">Overview of your AI agent system</p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {mockDashboardData.map((stat, idx) => (
            <div key={idx} className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
              <div className="flex items-center gap-3 mb-4">
                <div className={`w-10 h-10 bg-${stat.color}-100 rounded-lg flex items-center justify-center`}>
                  <span className={`text-${stat.color}-600 text-xl`}>{stat.icon}</span>
                </div>
                <h3 className="font-semibold text-gray-900">{stat.label}</h3>
              </div>
              <p className="text-2xl font-bold text-gray-900 mb-2">{stat.value}</p>
              <p className="text-sm text-gray-600">{stat.change}</p>
            </div>
          ))}
        </div>

        {/* Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Recent Activity */}
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Activity</h3>
            <div className="space-y-4">
              {mockRecentActivity.map((activity) => (
                <div key={activity.id} className="flex items-start gap-3 p-3 rounded-lg hover:bg-gray-50 transition-colors">
                  <div className={`w-2 h-2 rounded-full mt-2 ${
                    activity.type === 'success' ? 'bg-green-500' :
                    activity.type === 'info' ? 'bg-blue-500' :
                    'bg-yellow-500'
                  }`}></div>
                  <div className="flex-1">
                    <p className="text-sm text-gray-900">{activity.action}</p>
                    <p className="text-xs text-gray-500 mt-1">{activity.time}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Quick Actions */}
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
            <div className="grid grid-cols-2 gap-4">
              <button className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors text-left">
                <div className="text-2xl mb-2">🚀</div>
                <div className="font-medium text-gray-900">Deploy Agent</div>
                <div className="text-sm text-gray-500">Launch new AI agent</div>
              </button>
              
              <button className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors text-left">
                <div className="text-2xl mb-2">📊</div>
                <div className="font-medium text-gray-900">View Analytics</div>
                <div className="text-sm text-gray-500">System performance</div>
              </button>
              
              <button className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors text-left">
                <div className="text-2xl mb-2">⚙️</div>
                <div className="font-medium text-gray-900">Configuration</div>
                <div className="text-sm text-gray-500">System settings</div>
              </button>
              
              <button className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors text-left">
                <div className="text-2xl mb-2">💬</div>
                <div className="font-medium text-gray-900">Start Chat</div>
                <div className="text-sm text-gray-500">Interact with agents</div>
              </button>
            </div>
          </div>
        </div>

        {/* System Health Chart */}
        <div className="mt-8 bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">System Health</h3>
          <div className="h-64 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg flex items-center justify-center">
            <div className="text-center">
              <div className="text-4xl mb-4">📈</div>
              <p className="text-gray-600">Performance metrics visualization</p>
              <p className="text-sm text-gray-500 mt-2">Connect to your monitoring system</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// Placeholder Components
function ChatPage() {
  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold text-gray-900 mb-4">Chat Interface</h1>
      <div className="bg-white rounded-xl border border-gray-200 p-12 text-center">
        <div className="text-6xl mb-4">💬</div>
        <h3 className="text-xl font-semibold text-gray-900 mb-2">AI Chat Interface</h3>
        <p className="text-gray-600">Interactive chat with your AI agents</p>
      </div>
    </div>
  )
}

function AgentsPage() {
  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold text-gray-900 mb-4">Agent Management</h1>
      <div className="bg-white rounded-xl border border-gray-200 p-12 text-center">
        <div className="text-6xl mb-4">🤖</div>
        <h3 className="text-xl font-semibold text-gray-900 mb-2">Manage AI Agents</h3>
        <p className="text-gray-600">Configure, monitor and deploy your AI agents</p>
      </div>
    </div>
  )
}

function SettingsPage() {
  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold text-gray-900 mb-4">Settings</h1>
      <div className="bg-white rounded-xl border border-gray-200 p-12 text-center">
        <div className="text-6xl mb-4">⚙️</div>
        <h3 className="text-xl font-semibold text-gray-900 mb-2">System Configuration</h3>
        <p className="text-gray-600">Configure API keys, models and system preferences</p>
      </div>
    </div>
  )
}

// Main App Component
export default function ModernApp() {
  const [menuOpen, setMenuOpen] = useState(false)
  const [activeItem, setActiveItem] = useState('/')

  const renderPage = () => {
    switch (activeItem) {
      case '/': return <Dashboard />
      case '/chat': return <ChatPage />
      case '/agents': return <AgentsPage />
      case '/settings': return <SettingsPage />
      default: 
        const item = navigationItems.find(nav => nav.path === activeItem)
        return (
          <div className="p-8">
            <h1 className="text-3xl font-bold text-gray-900 mb-4">{item?.name || 'Page'}</h1>
            <div className="bg-white rounded-xl border border-gray-200 p-12 text-center">
              <div className="text-6xl mb-4">{item?.icon || '📄'}</div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">{item?.name} Module</h3>
              <p className="text-gray-600">This page is under development</p>
            </div>
          </div>
        )
    }
  }

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Mobile Menu Button */}
      <button
        className="sm:hidden fixed top-4 left-4 z-50 p-2 bg-white rounded-lg shadow-lg border border-gray-200"
        onClick={() => setMenuOpen(!menuOpen)}
        aria-label="Toggle menu"
      >
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>

      <ModernSidebar 
        open={menuOpen} 
        setOpen={setMenuOpen} 
        activeItem={activeItem} 
        setActiveItem={setActiveItem} 
      />
      
      <div className="flex-1 overflow-auto ml-0 sm:ml-0">
        {renderPage()}
      </div>
    </div>
  )
}
