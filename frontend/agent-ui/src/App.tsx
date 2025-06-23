import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import ChatPage from './pages/ChatPage'
import RoutingPage from './pages/RoutingPage'
import FeedbackPage from './pages/FeedbackPage'
import MonitoringPage from './pages/MonitoringPage'

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex h-screen">
        <Sidebar />
        <div className="flex-1 overflow-hidden">
          <Routes>
            <Route path="/" element={<ChatPage />} />
            <Route path="/routing" element={<RoutingPage />} />
            <Route path="/feedback" element={<FeedbackPage />} />
            <Route path="/monitoring" element={<MonitoringPage />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  )
}
