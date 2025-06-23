import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import ChatPage from './pages/ChatPage'
import RoutingPage from './pages/RoutingPage'
import FeedbackPage from './pages/FeedbackPage'
import MonitoringPage from './pages/MonitoringPage'
import AgentsPage from './pages/AgentsPage'
import TasksPage from './pages/TasksPage'
import SettingsPage from './pages/SettingsPage'
import MetricsPage from './pages/MetricsPage'
import AdminPage from './pages/AdminPage'
import DebugPage from './pages/DebugPage'

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex h-screen">
        <Sidebar />
        <div className="flex-1 overflow-hidden">
          <Routes>
            <Route path="/" element={<ChatPage />} />
            <Route path="/agents" element={<AgentsPage />} />
            <Route path="/tasks" element={<TasksPage />} />
            <Route path="/routing" element={<RoutingPage />} />
            <Route path="/feedback" element={<FeedbackPage />} />
            <Route path="/monitoring" element={<MonitoringPage />} />
            <Route path="/metrics" element={<MetricsPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/admin" element={<AdminPage />} />
            <Route path="/debug" element={<DebugPage />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  )
}
