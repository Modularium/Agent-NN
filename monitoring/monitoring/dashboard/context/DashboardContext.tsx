// src/context/DashboardContext.tsx
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { 
  SystemData, 
  SystemMetrics,
  Agent, 
  Model, 
  KnowledgeBase, 
  SecurityStatus, 
  SecurityEvent,
  TestResult, 
  LogEntry,
  ActiveTask
} from '../types/system';
import { fetchSystemData } from '../utils/api';
import useWebSocket from '../hooks/useWebSocket';
import { useNotification } from '../components/common/NotificationSystem';

interface DashboardContextType {
  systemData: SystemData | null;
  agents: Agent[];
  models: Model[];
  knowledgeBases: KnowledgeBase[];
  securityStatus: SecurityStatus | null;
  testResults: TestResult[];
  logs: LogEntry[];
  activeTab: string;
  setActiveTab: (tab: string) => void;
  loading: boolean;
  error: Error | null;
  refreshData: () => Promise<void>;
  lastUpdated: Date | null;
  isRealTimeEnabled: boolean;
  toggleRealTime: () => void;
  isRealTimeConnected: boolean;
}

const DashboardContext = createContext<DashboardContextType | undefined>(undefined);

export const DashboardProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [systemData, setSystemData] = useState<SystemData | null>(null);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [securityStatus, setSecurityStatus] = useState<SecurityStatus | null>(null);
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [activeTab, setActiveTab] = useState('system');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [isRealTimeEnabled, setIsRealTimeEnabled] = useState(true);
  
  const { addNotification } = useNotification();

  // WebSocket integration
  const handleMetricsUpdate = (metrics: SystemMetrics) => {
    if (systemData) {
      setSystemData({
        ...systemData,
        metrics,
        lastUpdated: new Date().toISOString()
      });
      setLastUpdated(new Date());
    }
  };
  
  const handleSecurityAlert = (event: SecurityEvent) => {
    if (securityStatus) {
      setSecurityStatus({
        ...securityStatus,
        events: [event, ...securityStatus.events.slice(0, 9)] // Keep last 10 events
      });
    }
    
    // Add to logs
    const newLog: LogEntry = {
      level: event.severity === 'high' ? 'ERROR' : 
             event.severity === 'medium' ? 'WARNING' : 'INFO',
      timestamp: event.timestamp,
      message: `Security event: ${event.details}`,
      source: 'Security'
    };
    
    setLogs(prev => [newLog, ...prev]);
    setLastUpdated(new Date());
  };
  
  const handleTaskUpdate = (task: ActiveTask) => {
    if (systemData) {
      // Find if task already exists and update it, or add as new
      const existingTaskIndex = systemData.activeTasks.findIndex(t => t.id === task.id);
      
      const updatedTasks = [...systemData.activeTasks];
      if (existingTaskIndex >= 0) {
        updatedTasks[existingTaskIndex] = task;
      } else {
        updatedTasks.push(task);
      }
      
      // Filter out completed tasks after a while (or keep them at the end)
      const filteredTasks = updatedTasks
        .filter(t => t.status !== 'completed' || t.id === task.id)
        .sort((a, b) => {
          // Sort by status (running first, then queued, then completed)
          const statusOrder = { running: 0, queued: 1, completed: 2, failed: 3 };
          return statusOrder[a.status] - statusOrder[b.status];
        });
      
      setSystemData({
        ...systemData,
        activeTasks: filteredTasks,
        lastUpdated: new Date().toISOString()
      });
      
      // Update agent last activity if applicable
      const relatedAgent = agents.find(a => a.name === task.agent);
      if (relatedAgent) {
        const updatedAgents = agents.map(a => 
          a.name === task.agent 
            ? { ...a, lastActive: 'Just now' } 
            : a
        );
        setAgents(updatedAgents);
      }
      
      setLastUpdated(new Date());
    }
  };
  
  const handleAgentStatusChange = (agentName: string, status: string) => {
    const updatedAgents = agents.map(agent => 
      agent.name === agentName 
        ? { ...agent, status: status as 'active' | 'idle' | 'error' } 
        : agent
    );
    
    setAgents(updatedAgents);
    setLastUpdated(new Date());
  };
  
  // Connect to WebSocket for real-time updates
  const { isConnected: isRealTimeConnected } = useWebSocket({
    autoConnect: isRealTimeEnabled,
    onMetricsUpdate: handleMetricsUpdate,
    onSecurityAlert: handleSecurityAlert,
    onTaskUpdate: handleTaskUpdate,
    onAgentStatusChange: handleAgentStatusChange,
    showNotifications: true
  });
  
  // Toggle real-time updates
  const toggleRealTime = () => {
    setIsRealTimeEnabled(!isRealTimeEnabled);
    
    // Show notification
    addNotification({
      type: 'info',
      title: !isRealTimeEnabled ? 'Real-time Updates Enabled' : 'Real-time Updates Disabled',
      message: !isRealTimeEnabled 
        ? 'You will now receive real-time updates.' 
        : 'Real-time updates are now disabled. Refresh manually for new data.',
      duration: 3000
    });
  };

  // Fetch data function
  const refreshData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const data = await fetchSystemData();
      
      setSystemData(data.systemData);
      setAgents(data.agents);
      setModels(data.models);
      setKnowledgeBases(data.knowledgeBases);
      setSecurityStatus(data.securityStatus);
      setTestResults(data.testResults);
      setLogs(data.logs);
      setLastUpdated(new Date());
      
      setLoading(false);
      return data;
    } catch (err) {
      setError(err instanceof Error ? err : new Error('An unknown error occurred'));
      setLoading(false);
      throw err;
    }
  };

  // Initial data fetch
  useEffect(() => {
    refreshData().catch(error => {
      console.error('Error fetching initial data:', error);
    });
  }, []);

  // Context value
  const value = {
    systemData,
    agents,
    models,
    knowledgeBases,
    securityStatus,
    testResults,
    logs,
    activeTab,
    setActiveTab,
    loading,
    error,
    refreshData,
    lastUpdated,
    isRealTimeEnabled,
    toggleRealTime,
    isRealTimeConnected
  };

  return (
    <DashboardContext.Provider value={value}>
      {children}
    </DashboardContext.Provider>
  );
};

export const useDashboard = () => {
  const context = useContext(DashboardContext);
  if (context === undefined) {
    throw new Error('useDashboard must be used within a DashboardProvider');
  }
  return context;
};
    
    // Set up auto-refresh (optional)
    const interval = setInterval(() => {
      refreshData();
    }, 30000); // Refresh every 30 seconds
    
    return () => clearInterval(interval);
  }, []);

  const value = {
    systemData,
    agents,
    models,
    knowledgeBases,
    securityStatus,
    testResults,
    logs,
    activeTab,
    setActiveTab,
    loading,
    error,
    refreshData,
    lastUpdated
  };

  return (
    <DashboardContext.Provider value={value}>
      {children}
    </DashboardContext.Provider>
  );
};

export const useDashboard = () => {
  const context = useContext(DashboardContext);
  if (context === undefined) {
    throw new Error('useDashboard must be used within a DashboardProvider');
  }
  return context;
};

// monitoring/dashboard/context/ThemeContext.tsx
import React, { createContext, useContext, useState, ReactNode, useEffect } from 'react';

type ThemeMode = 'light' | 'dark';

interface ThemeContextType {
  themeMode: ThemeMode;
  toggleTheme: () => void;
  setThemeMode: (mode: ThemeMode) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const ThemeProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [themeMode, setThemeMode] = useState<ThemeMode>('light');

  useEffect(() => {
    // Check local storage or system preference
    const storedTheme = localStorage.getItem('themeMode') as ThemeMode | null;
    if (storedTheme) {
      setThemeMode(storedTheme);
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      setThemeMode('dark');
    }
  }, []);

  useEffect(() => {
    // Update body class and store preference
    document.body.classList.toggle('dark-theme', themeMode === 'dark');
    localStorage.setItem('themeMode', themeMode);
  }, [themeMode]);

  const toggleTheme = () => {
    setThemeMode(prevMode => (prevMode === 'light' ? 'dark' : 'light'));
  };

  const value = {
    themeMode,
    toggleTheme,
    setThemeMode
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

// monitoring/dashboard/context/AuthContext.tsx
import React, { createContext, useContext, useState, ReactNode, useEffect } from 'react';

interface User {
  username: string;
  role: string;
  permissions: string[];
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  loading: boolean;
  error: string | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Check for stored auth token
    const checkAuth = async () => {
      try {
        const token = localStorage.getItem('authToken');
        if (token) {
          // Validate token with API (mock for now)
          setUser({
            username: 'admin',
            role: 'administrator',
            permissions: ['read', 'write', 'manage']
          });
        }
      } catch (err) {
        console.error('Auth validation error:', err);
      } finally {
        setLoading(false);
      }
    };
    
    checkAuth();
  }, []);

  const login = async (username: string, password: string) => {
    setLoading(true);
    setError(null);
    
    try {
      // Mock API call for now
      if (username === 'admin' && password === 'password') {
        const user = {
          username: 'admin',
          role: 'administrator',
          permissions: ['read', 'write', 'manage']
        };
        
        localStorage.setItem('authToken', 'mock-token');
        setUser(user);
      } else {
        throw new Error('Invalid credentials');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    localStorage.removeItem('authToken');
    setUser(null);
  };

  const value = {
    user,
    isAuthenticated: !!user,
    login,
    logout,
    loading,
    error
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
