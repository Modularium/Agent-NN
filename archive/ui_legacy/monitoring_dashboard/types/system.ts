// monitoring/dashboard/types/system.ts
export interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  gpu_usage: number;
  disk_usage: number;
  active_agents: number;
  task_queue_size: number;
  total_tasks_completed: number;
  avg_response_time: number;
}

export interface SystemComponent {
  name: string;
  status: 'online' | 'offline' | 'degraded';
  version: string;
  lastUpdated: string;
}

export interface ActiveTask {
  id: string;
  type: string;
  agent: string;
  status: 'running' | 'completed' | 'queued' | 'failed';
  duration: string;
}

export interface SystemData {
  metrics: SystemMetrics;
  components: SystemComponent[];
  activeTasks: ActiveTask[];
  lastUpdated: string;
}

// monitoring/dashboard/types/agent.ts
export interface Agent {
  name: string;
  domain: string;
  status: 'active' | 'idle' | 'error';
  tasks: number;
  successRate: number;
  avgResponse: number;
  lastActive: string;
}

export interface AgentCapability {
  name: string;
  description: string;
  successRate: number;
}

export interface AgentDetail extends Agent {
  capabilities: AgentCapability[];
  knowledgeBase: string;
  model: string;
  createdAt: string;
}

export interface AgentTemplate {
  id: string;
  name: string;
  domain: string;
  description: string;
  baseModel: string;
  knowledgeBase: string;
}

// monitoring/dashboard/types/model.ts
export interface Model {
  name: string;
  type: string;
  source: string;
  version: string;
  status: 'active' | 'inactive' | 'loading';
  requests: number;
  latency: number;
}

export interface ModelPerformance {
  name: string;
  responseTime: number[];
  successRate: number[];
  timestamps: string[];
}

// monitoring/dashboard/types/metrics.ts
export interface TimeSeriesData {
  timestamp: string;
  value: number;
}

export interface MetricData {
  name: string;
  current: number;
  history: TimeSeriesData[];
  status: 'normal' | 'warning' | 'critical';
  unit: string;
  threshold?: {
    warning: number;
    critical: number;
  };
}

// monitoring/dashboard/types/knowledge.ts
export interface KnowledgeBase {
  name: string;
  documents: number;
  lastUpdated: string;
  size: string;
  status: 'active' | 'updating' | 'error';
}

export interface Document {
  id: string;
  title: string;
  type: string;
  size: number;
  uploadedAt: string;
  lastAccessed: string;
}

// monitoring/dashboard/types/security.ts
export interface SecurityEvent {
  type: string;
  timestamp: string;
  details: string;
  severity: 'low' | 'medium' | 'high';
}

export interface SecurityStatus {
  overall: 'secure' | 'warning' | 'breach';
  lastScan: string;
  vulnerabilities: {
    high: number;
    medium: number;
    low: number;
  };
  events: SecurityEvent[];
}

// monitoring/dashboard/types/test.ts
export interface TestResult {
  id: string;
  name: string;
  status: 'completed' | 'in-progress' | 'failed';
  variants: number;
  winner: string;
  improvement: string;
}

export interface TestVariant {
  name: string;
  description: string;
  metrics: {
    [key: string]: number;
  };
}

export interface TestDetail {
  id: string;
  name: string;
  status: 'completed' | 'in-progress' | 'failed';
  variants: TestVariant[];
  startDate: string;
  endDate: string;
}

// monitoring/dashboard/types/logs.ts
export interface LogEntry {
  level: 'INFO' | 'WARNING' | 'ERROR' | 'DEBUG';
  timestamp: string;
  message: string;
  source?: string;
}

// monitoring/dashboard/types/settings.ts
export interface SystemSettings {
  name: string;
  timezone: string;
  language: string;
  logLevel: string;
  logRetention: number;
  automaticUpdates: boolean;
  usageAnalytics: boolean;
  emailNotifications: boolean;
}
