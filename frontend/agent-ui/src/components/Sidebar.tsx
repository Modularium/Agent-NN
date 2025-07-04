import { useState } from 'react'

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

export default function ModernSidebar() {
  const [open, setOpen] = useState(false)
  const [activeItem, setActiveItem] = useState('/')

  const linkClass = (isActive: boolean) =>
    `flex items-center gap-3 px-4 py-3 mx-2 rounded-xl transition-all duration-200 hover:bg-gradient-to-r hover:from-blue-50 hover:to-indigo-50 hover:shadow-sm group cursor-pointer ${
      isActive 
        ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-lg shadow-blue-500/25' 
        : 'text-gray-700 hover:text-blue-600'
    }`

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Mobile Menu Button */}
      <button
        className="sm:hidden fixed top-4 left-4 z-50 p-2 bg-white rounded-lg shadow-lg border border-gray-200"
        onClick={() => setOpen(!open)}
        aria-label="Toggle menu"
      >
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>

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

      {/* Main Content Area */}
      <div className="flex-1 overflow-hidden ml-0 sm:ml-0">
        <div className="p-8 pt-16 sm:pt-8">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-3xl font-bold text-gray-900 mb-2">
              {navigationItems.find(item => item.path === activeItem)?.name || 'Dashboard'}
            </h2>
            <p className="text-gray-600 mb-8">
              Manage your AI agents and monitor system performance
            </p>
            
            {/* Sample Content */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                    <span className="text-blue-600 text-xl">🤖</span>
                  </div>
                  <h3 className="font-semibold text-gray-900">Active Agents</h3>
                </div>
                <p className="text-2xl font-bold text-gray-900 mb-2">12</p>
                <p className="text-sm text-gray-600">+2 from yesterday</p>
              </div>

              <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                    <span className="text-green-600 text-xl">✅</span>
                  </div>
                  <h3 className="font-semibold text-gray-900">Completed Tasks</h3>
                </div>
                <p className="text-2xl font-bold text-gray-900 mb-2">847</p>
                <p className="text-sm text-gray-600">94% success rate</p>
              </div>

              <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
                    <span className="text-purple-600 text-xl">⚡</span>
                  </div>
                  <h3 className="font-semibold text-gray-900">Response Time</h3>
                </div>
                <p className="text-2xl font-bold text-gray-900 mb-2">0.3s</p>
                <p className="text-sm text-gray-600">Average latency</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
