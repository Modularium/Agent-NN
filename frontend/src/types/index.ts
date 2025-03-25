export interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
  error?: boolean
  metadata?: {
    agent?: string
    executionTime?: number
  }
}

export interface Agent {
  id: string
  name: string
  domain: string
  totalTasks: number
  successRate: number
  description: string
  avgExecutionTime: number
  knowledgeBase: {
    documentsCount: number
  }
}

export interface TaskEvent {
  type: string
  timestamp: string
  description: string
  status: 'success' | 'error' | 'warning' | 'info'
  metadata?: {
    agent?: string
    executionTime?: number
  }
}

export interface Task {
  id: string
  description: string
  status: 'completed' | 'in_progress' | 'failed' | 'pending'
  timestamp: string
  agent: string
  executionTime: number
  events: TaskEvent[]
  result?: string
}

export interface Settings {
  llmBackend: string
  apiKey: string
  maxTokens: number
  temperature: number
  enableLocalModels: boolean
  enableLogging: boolean
  language: string
}