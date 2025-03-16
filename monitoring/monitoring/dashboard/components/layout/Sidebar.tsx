// Sidebar.tsx - Navigation sidebar for the dashboard
import React from 'react';
import { 
  Server, 
  Users, 
  Cpu, 
  Database, 
  BarChart, 
  Shield, 
  GitBranch, 
  Settings, 
  AlertTriangle, 
  Book,
  Menu,
  X
} from 'lucide-react';
import { useDashboard } from '../../context/DashboardContext';

interface SidebarProps {
  collapsed: boolean;
  toggleCollapse: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ collapsed, toggleCollapse }) => {
  const { activeTab, setActiveTab } = useDashboard();
  
  // Navigation items configuration
  const navItems = [
    { id: 'system', label: 'System Overview', icon: <Server size={20} /> },
    { id: 'agents', label: 'Agents', icon: <Users size={20} /> },
    { id: 'models', label: 'Models', icon: <Cpu size={20} /> },
    { id: 'knowledge', label: 'Knowledge Bases', icon: <Database size={20} /> },
    { id: 'monitoring', label: 'Monitoring', icon: <BarChart size={20} /> },
    { id: 'security', label: 'Security', icon: <Shield size={20} /> },
    { id: 'testing', label: 'A/B Testing', icon: <GitBranch size={20} /> },
    { id: 'settings', label: 'Settings', icon: <Settings size={20} /> },
    { id: 'logs', label: 'Logs & Alerts', icon: <AlertTriangle size={20} /> },
    { id: 'docs', label: 'Documentation', icon: <Book size={20} /> },
  ];
  
  return (
    <aside className={`${collapsed ? 'w-16' : 'w-64'} bg-white dark:bg-gray-800 shadow-md transition-width duration-300 flex flex-col`}>
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
        {!collapsed && <h2 className="text-lg font-semibold text-gray-800 dark:text-white">Navigation</h2>}
        <button 
          className="text-gray-500 hover:text-gray-700 dark:text-gray-300 dark:hover:text-white"
          onClick={toggleCollapse}
        >
          {collapsed ? <Menu size={20} /> : <X size={20} />}
        </button>
      </div>
      
      <nav className="flex-1 overflow-y-auto p-2">
        <ul className="space-y-1">
          {navItems.map(item => (
            <li key={item.id}>
              <button 
                className={`flex items-center ${collapsed ? 'justify-center' : 'space-x-3'} w-full p-3 rounded-md ${activeTab === item.id ? 'bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300' : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300'}`}
                onClick={() => setActiveTab(item.id)}
              >
                <span>{item.icon}</span>
                {!collapsed && <span>{item.label}</span>}
              </button>
            </li>
          ))}
        </ul>
      </nav>
      
      <div className="p-4 border-t border-gray-200 dark:border-gray-700">
        <div className={`flex ${collapsed ? 'justify-center' : 'items-center space-x-3'}`}>
          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
          {!collapsed && <span className="text-sm text-gray-600 dark:text-gray-400">System Online</span>}
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
