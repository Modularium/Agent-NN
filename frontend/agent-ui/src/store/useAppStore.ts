// src/store/useAppStore.ts
import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { immer } from 'zustand/middleware/immer'

export interface User {
  id: string
  name: string
  email: string
  avatar?: string
  role: string
  permissions: string[]
}

export interface Notification {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message: string
  timestamp: Date
  read: boolean
  actions?: Array<{
    label: string
    action: () => void
    variant?: 'primary' | 'secondary'
  }>
}

export interface Agent {
  id: string
  name: string
  domain: string
  status: 'active' | 'idle' | 'error' | 'maintenance'
  version: string
  description: string
  capabilities: string[]
  metrics: {
    totalTasks: number
    successRate: number
    avgResponseTime: number
    lastActive: Date
  }
  configuration: Record<string, any>
}

export interface Task {
  id: string
  title: string
  description: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  priority: 'low' | 'medium' | 'high' | 'urgent'
  type: string
  agentId: string
  assignedAt: Date
  startedAt?: Date
  completedAt?: Date
  progress: number
  metadata: Record<string, any>
  error?: {
    code: string
    message: string
    stack?: string
  }
}

export interface AppSettings {
  theme: 'light' | 'dark' | 'auto'
  language: string
  notifications: {
    enabled: boolean
    sound: boolean
    desktop: boolean
  }
  ui: {
    sidebarCollapsed: boolean
    density: 'compact' | 'comfortable' | 'spacious'
    animations: boolean
  }
  api: {
    baseUrl: string
    timeout: number
    retries: number
  }
}

interface AppState {
  // UI State
  sidebarOpen: boolean
  loading: boolean
  online: boolean
  
  // User
  user: User | null
  authenticated: boolean
  
  // Data
  agents: Agent[]
  tasks: Task[]
  notifications: Notification[]
  
  // Settings
  settings: AppSettings
  
  // Error state
  error: string | null
}

interface AppActions {
  // UI Actions
  setSidebarOpen: (open: boolean) => void
  setLoading: (loading: boolean) => void
  setOnline: (online: boolean) => void
  
  // User Actions
  setUser: (user: User | null) => void
  setAuthenticated: (authenticated: boolean) => void
  logout: () => void
  
  // Agent Actions
  setAgents: (agents: Agent[]) => void
  addAgent: (agent: Agent) => void
  updateAgent: (id: string, updates: Partial<Agent>) => void
  removeAgent: (id: string) => void
  
  // Task Actions
  setTasks: (tasks: Task[]) => void
  addTask: (task: Task) => void
  updateTask: (id: string, updates: Partial<Task>) => void
  removeTask: (id: string) => void
  
  // Notification Actions
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => void
  markNotificationRead: (id: string) => void
  removeNotification: (id: string) => void
  clearAllNotifications: () => void
  
  // Settings Actions
  updateSettings: (updates: Partial<AppSettings>) => void
  resetSettings: () => void
  
  // Error Actions
  setError: (error: string | null) => void
  clearError: () => void
}

const defaultSettings: AppSettings = {
  theme: 'light',
  language: 'en',
  notifications: {
    enabled: true,
    sound: true,
    desktop: true
  },
  ui: {
    sidebarCollapsed: false,
    density: 'comfortable',
    animations: true
  },
  api: {
    baseUrl: import.meta.env.VITE_API_URL || 'http://localhost:8000',
    timeout: 30000,
    retries: 3
  }
}

export const useAppStore = create<AppState & AppActions>()(
  persist(
    immer((set, get) => ({
      // Initial State
      sidebarOpen: false,
      loading: false,
      online: navigator.onLine,
      user: null,
      authenticated: false,
      agents: [],
      tasks: [],
      notifications: [],
      settings: defaultSettings,
      error: null,

      // UI Actions
      setSidebarOpen: (open) =>
        set((state) => {
          state.sidebarOpen = open
        }),

      setLoading: (loading) =>
        set((state) => {
          state.loading = loading
        }),

      setOnline: (online) =>
        set((state) => {
          state.online = online
        }),

      // User Actions
      setUser: (user) =>
        set((state) => {
          state.user = user
          state.authenticated = !!user
        }),

      setAuthenticated: (authenticated) =>
        set((state) => {
          state.authenticated = authenticated
          if (!authenticated) {
            state.user = null
          }
        }),

      logout: () =>
        set((state) => {
          state.user = null
          state.authenticated = false
          state.notifications = []
          // Keep settings but reset other data
        }),

      // Agent Actions
      setAgents: (agents) =>
        set((state) => {
          state.agents = agents
        }),

      addAgent: (agent) =>
        set((state) => {
          state.agents.push(agent)
        }),

      updateAgent: (id, updates) =>
        set((state) => {
          const index = state.agents.findIndex(a => a.id === id)
          if (index !== -1) {
            state.agents[index] = { ...state.agents[index], ...updates }
          }
        }),

      removeAgent: (id) =>
        set((state) => {
          state.agents = state.agents.filter(a => a.id !== id)
        }),

      // Task Actions
      setTasks: (tasks) =>
        set((state) => {
          state.tasks = tasks
        }),

      addTask: (task) =>
        set((state) => {
          state.tasks.push(task)
        }),

      updateTask: (id, updates) =>
        set((state) => {
          const index = state.tasks.findIndex(t => t.id === id)
          if (index !== -1) {
            state.tasks[index] = { ...state.tasks[index], ...updates }
          }
        }),

      removeTask: (id) =>
        set((state) => {
          state.tasks = state.tasks.filter(t => t.id !== id)
        }),

      // Notification Actions
      addNotification: (notification) =>
        set((state) => {
          const newNotification: Notification = {
            ...notification,
            id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
            timestamp: new Date(),
            read: false
          }
          state.notifications.unshift(newNotification)
          
          // Keep only last 50 notifications
          if (state.notifications.length > 50) {
            state.notifications = state.notifications.slice(0, 50)
          }
        }),

      markNotificationRead: (id) =>
        set((state) => {
          const notification = state.notifications.find(n => n.id === id)
          if (notification) {
            notification.read = true
          }
        }),

      removeNotification: (id) =>
        set((state) => {
          state.notifications = state.notifications.filter(n => n.id !== id)
        }),

      clearAllNotifications: () =>
        set((state) => {
          state.notifications = []
        }),

      // Settings Actions
      updateSettings: (updates) =>
        set((state) => {
          state.settings = { ...state.settings, ...updates }
        }),

      resetSettings: () =>
        set((state) => {
          state.settings = defaultSettings
        }),

      // Error Actions
      setError: (error) =>
        set((state) => {
          state.error = error
        }),

      clearError: () =>
        set((state) => {
          state.error = null
        })
    })),
    {
      name: 'agent-nn-store',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        settings: state.settings,
        user: state.user,
        authenticated: state.authenticated
      })
    }
  )
)

// Selectors
export const useUser = () => useAppStore((state) => state.user)
export const useAuthenticated = () => useAppStore((state) => state.authenticated)
export const useAgents = () => useAppStore((state) => state.agents)
export const useTasks = () => useAppStore((state) => state.tasks)
export const useNotifications = () => useAppStore((state) => state.notifications)
export const useSettings = () => useAppStore((state) => state.settings)
export const useLoading = () => useAppStore((state) => state.loading)
export const useError = () => useAppStore((state) => state.error)
