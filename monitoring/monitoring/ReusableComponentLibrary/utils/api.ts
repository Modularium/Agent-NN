// monitoring/dashboard/utils/api.ts
import { 
  SystemData, 
  Agent, 
  Model, 
  KnowledgeBase, 
  SecurityStatus, 
  TestResult, 
  LogEntry 
} from '../types';

// API base URL - can be configured from environment
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

// Generic fetch wrapper with error handling
async function fetchWithError<T>(url: string, options = {}): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options as any).headers,
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    throw new Error(
      errorData?.message || `API request failed with status ${response.status}`
    );
  }

  return response.json();
}

// Mock data generator for development
function generateMockData() {
  return {
    systemData: {
      metrics: {
        cpu_usage: Math.floor(Math.random() * 100),
        memory_usage: Math.floor(Math.random() * 100),
        gpu_usage: Math.floor(Math.random() * 100),
        disk_usage: Math.floor(Math.random() * 100),
        active_agents: 5,
        task_queue_size: Math.floor(Math.random() * 20),
        total_tasks_completed: 2435,
        avg_response_time: 1.2 + Math.random()
      },
      components: [
        { name: 'Supervisor Agent', status: 'online', version: '2.1.0', lastUpdated: '2 hours ago' },
        { name: 'MLflow Integration', status: 'online', version: '1.4.2', lastUpdated: '1 day ago' },
        { name: 'Vector Store', status: 'online', version: '3.0.1', lastUpdated: '3 days ago' },
        { name: 'Cache Manager', status: 'online', version: '1.2.5', lastUpdated: '5 days ago' }
      ],
      activeTasks: [
        { id: 'T-1254', type: 'Analysis', agent: 'Finance', status: 'running', duration: '24s' },
        { id: 'T-1253', type: 'Research', agent: 'Web', status: 'completed', duration: '3m 12s' },
        { id: 'T-1252', type: 'Code', agent: 'Tech', status: 'queued', duration: '-' }
      ],
      lastUpdated: new Date().toISOString()
    },
    agents: [
      { name: 'Finance Agent', domain: 'Finance', status: 'active', tasks: 1245, successRate: 92.5, avgResponse: 1.2, lastActive: 'Just now' },
      { name: 'Tech Agent', domain: 'Technology', status: 'active', tasks: 2153, successRate: 94.1, avgResponse: 0.9, lastActive: '2 minutes ago' },
      { name: 'Marketing Agent', domain: 'Marketing', status: 'active', tasks: 987, successRate: 88.3, avgResponse: 1.5, lastActive: '5 minutes ago' },
      { name: 'Web Agent', domain: 'Web', status: 'active', tasks: 1856, successRate: 91.2, avgResponse: 1.1, lastActive: '10 minutes ago' },
      { name: 'Research Agent', domain: 'Research', status: 'idle', tasks: 654, successRate: 89.7, avgResponse: 2.3, lastActive: '1 hour ago' }
    ],
    models: [
      { name: 'gpt-4', type: 'LLM', source: 'OpenAI', version: 'v1.0', status: 'active', requests: 32145, latency: 1.2 },
      { name: 'claude-3', type: 'LLM', source: 'Anthropic', version: 'v1.0', status: 'active', requests: 18921, latency: 1.5 },
      { name: 'llama-3', type: 'LLM', source: 'Local', version: 'v1.0', status: 'active', requests: 8752, latency: 2.1 }
    ],
    knowledgeBases: [
      { name: 'Finance KB', documents: 1245, lastUpdated: '2 hours ago', size: '2.4 GB', status: 'active' },
      { name: 'Tech KB', documents: 3567, lastUpdated: '1 day ago', size: '5.2 GB', status: 'active' },
      { name: 'Marketing KB', documents: 982, lastUpdated: '3 days ago', size: '1.8 GB', status: 'active' },
      { name: 'General KB', documents: 4521, lastUpdated: '5 days ago', size: '8.7 GB', status: 'active' }
    ],
    securityStatus: {
      overall: 'secure',
      lastScan: '2025-03-16T08:00:00Z',
      vulnerabilities: { high: 0, medium: 2, low: 5 },
      events: [
        { type: 'Authentication', timestamp: '2025-03-16T14:23:45Z', details: 'Successful login: admin', severity: 'low' },
        { type: 'Rate Limit', timestamp: '2025-03-16T14:10:30Z', details: 'Rate limit exceeded: 192.168.1.105', severity: 'medium' },
        { type: 'Input Validation', timestamp: '2025-03-16T13:45:15Z', details: 'Suspicious input detected and sanitized', severity: 'medium' }
      ]
    },
    testResults: [
      { id: 'test-001', name: 'Prompt Optimization', status: 'completed', variants: 2, winner: 'Variant B', improvement: '+12.5%' },
      { id: 'test-002', name: 'Model Comparison', status: 'in-progress', variants: 3, winner: '-', improvement: '-' },
      { id: 'test-003', name: 'Knowledge Source', status: 'completed', variants: 2, winner: 'Variant A', improvement: '+5.2%' }
    ],
    logs: [
      { level: 'INFO', timestamp: '2025-03-16T14:23:45Z', message: 'Agent "Finance" created successfully' },
      { level: 'WARNING', timestamp: '2025-03-16T14:22:30Z', message: 'High memory usage detected (78%)' },
      { level: 'ERROR', timestamp: '2025-03-16T14:20:15Z', message: 'Failed to load model "mistral-7b" - CUDA out of memory' },
      { level: 'INFO', timestamp: '2025-03-16T14:19:45Z', message: 'System started successfully' }
    ]
  };
}

// Main API functions
export async function fetchSystemData() {
  // For development, use mock data
  if (process.env.REACT_APP_USE_MOCK_DATA === 'true') {
    return generateMockData();
  }
  
  try {
    return await fetchWithError(`${API_BASE_URL}/system/dashboard-data`);
  } catch (error) {
    console.error('Error fetching system data:', error);
    throw error;
  }
}

export async function fetchAgentDetails(agentName: string) {
  try {
    return await fetchWithError(`${API_BASE_URL}/agents/${agentName}`);
  } catch (error) {
    console.error(`Error fetching agent details for ${agentName}:`, error);
    throw error;
  }
}

export async function createAgent(agentData: Partial<Agent>) {
  try {
    return await fetchWithError(`${API_BASE_URL}/agents`, {
      method: 'POST',
      body: JSON.stringify(agentData)
    });
  } catch (error) {
    console.error('Error creating agent:', error);
    throw error;
  }
}

export async function updateAgentStatus(agentName: string, status: 'active' | 'idle') {
  try {
    return await fetchWithError(`${API_BASE_URL}/agents/${agentName}/status`, {
      method: 'PUT',
      body: JSON.stringify({ status })
    });
  } catch (error) {
    console.error(`Error updating agent status for ${agentName}:`, error);
    throw error;
  }
}

export async function fetchLogs(filters: { level?: string; limit?: number; offset?: number } = {}) {
  const queryParams = new URLSearchParams();
  
  if (filters.level) queryParams.append('level', filters.level);
  if (filters.limit) queryParams.append('limit', filters.limit.toString());
  if (filters.offset) queryParams.append('offset', filters.offset.toString());
  
  const queryString = queryParams.toString() ? `?${queryParams.toString()}` : '';
  
  try {
    return await fetchWithError<LogEntry[]>(`${API_BASE_URL}/logs${queryString}`);
  } catch (error) {
    console.error('Error fetching logs:', error);
    throw error;
  }
}

// monitoring/dashboard/utils/formatters.ts
/**
 * Format a datetime string with relative time
 */
export function formatRelativeTime(timestamp: string): string {
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  
  if (diffSec < 60) {
    return 'Just now';
  }
  
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) {
    return `${diffMin} minute${diffMin !== 1 ? 's' : ''} ago`;
  }
  
  const diffHours = Math.floor(diffMin / 60);
  if (diffHours < 24) {
    return `${diffHours} hour${diffHours !== 1 ? 's' : ''} ago`;
  }
  
  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 30) {
    return `${diffDays} day${diffDays !== 1 ? 's' : ''} ago`;
  }
  
  // For older dates, return the actual date
  return date.toLocaleDateString();
}

/**
 * Format a number with appropriate units (K, M, B)
 */
export function formatNumber(num: number): string {
  if (num >= 1_000_000_000) {
    return (num / 1_000_000_000).toFixed(1) + 'B';
  }
  if (num >= 1_000_000) {
    return (num / 1_000_000).toFixed(1) + 'M';
  }
  if (num >= 1_000) {
    return (num / 1_000).toFixed(1) + 'K';
  }
  return num.toString();
}

/**
 * Format percentage values
 */
export function formatPercentage(value: number): string {
  return `${value.toFixed(1)}%`;
}

/**
 * Format time duration (seconds, minutes, hours)
 */
export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  }
  
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  
  if (minutes < 60) {
    return `${minutes}m ${Math.round(remainingSeconds)}s`;
  }
  
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  
  return `${hours}h ${remainingMinutes}m`;
}

/**
 * Format file size (B, KB, MB, GB)
 */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  
  if (bytes < 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

/**
 * Determine status color
 */
export function getStatusColor(status: string): string {
  switch (status.toLowerCase()) {
    case 'active':
    case 'online':
    case 'completed':
    case 'secure':
    case 'success':
      return 'green';
    case 'warning':
    case 'degraded':
    case 'in-progress':
    case 'running':
      return 'yellow';
    case 'error':
    case 'offline':
    case 'failed':
    case 'breach':
      return 'red';
    case 'idle':
    case 'queued':
    case 'inactive':
      return 'gray';
    default:
      return 'blue';
  }
}

// monitoring/dashboard/utils/chartHelpers.ts
import { TimeSeriesData } from '../types/metrics';

/**
 * Process time series data for charts
 * @param data Raw time series data
 * @param timeWindow Optional time window in milliseconds
 */
export function processTimeSeriesData(
  data: TimeSeriesData[],
  timeWindow?: number
): {
  labels: string[];
  values: number[];
} {
  // Apply time window filter if specified
  let filteredData = data;
  if (timeWindow) {
    const cutoff = Date.now() - timeWindow;
    filteredData = data.filter(point => new Date(point.timestamp).getTime() > cutoff);
  }

  // Sort by timestamp
  filteredData.sort((a, b) => 
    new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
  );

  // Extract labels and values
  return {
    labels: filteredData.map(point => {
      const date = new Date(point.timestamp);
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }),
    values: filteredData.map(point => point.value)
  };
}

/**
 * Generate chart colors based on theme
 */
export function getChartColors(isDarkMode: boolean): {
  backgroundColor: string[];
  borderColor: string[];
  gridColor: string;
  textColor: string;
} {
  if (isDarkMode) {
    return {
      backgroundColor: [
        'rgba(79, 70, 229, 0.2)',
        'rgba(16, 185, 129, 0.2)',
        'rgba(245, 158, 11, 0.2)',
        'rgba(239, 68, 68, 0.2)',
        'rgba(99, 102, 241, 0.2)'
      ],
      borderColor: [
        'rgba(79, 70, 229, 1)',
        'rgba(16, 185, 129, 1)',
        'rgba(245, 158, 11, 1)',
        'rgba(239, 68, 68, 1)',
        'rgba(99, 102, 241, 1)'
      ],
      gridColor: 'rgba(255, 255, 255, 0.1)',
      textColor: 'rgba(255, 255, 255, 0.7)'
    };
  }
  
  return {
    backgroundColor: [
      'rgba(79, 70, 229, 0.2)',
      'rgba(16, 185, 129, 0.2)',
      'rgba(245, 158, 11, 0.2)',
      'rgba(239, 68, 68, 0.2)',
      'rgba(99, 102, 241, 0.2)'
    ],
    borderColor: [
      'rgba(79, 70, 229, 1)',
      'rgba(16, 185, 129, 1)',
      'rgba(245, 158, 11, 1)',
      'rgba(239, 68, 68, 1)',
      'rgba(99, 102, 241, 1)'
    ],
    gridColor: 'rgba(0, 0, 0, 0.1)',
    textColor: 'rgba(0, 0, 0, 0.7)'
  };
}

/**
 * Get chart gradient for area charts
 */
export function createChartGradient(
  ctx: CanvasRenderingContext2D,
  color: string,
  isDarkMode: boolean
): CanvasGradient {
  const gradient = ctx.createLinearGradient(0, 0, 0, 400);
  gradient.addColorStop(0, `${color}40`); // 25% opacity
  gradient.addColorStop(1, isDarkMode ? 'rgba(17, 24, 39, 0.01)' : 'rgba(255, 255, 255, 0.01)');
  return gradient;
}

// monitoring/dashboard/utils/validators.ts
/**
 * Validate email format
 */
export function validateEmail(email: string): boolean {
  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return re.test(email);
}

/**
 * Validate URL format
 */
export function validateURL(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

/**
 * Validate numeric range
 */
export function validateNumericRange(
  value: number,
  min: number,
  max: number
): boolean {
  return value >= min && value <= max;
}
