// src/utils/websocketService.ts
import { io, Socket } from 'socket.io-client';
import { SystemMetrics, SecurityEvent, ActiveTask } from '../types/system';

// Event types for WebSocket communication
type WebSocketEvent = 
  | 'connect'
  | 'disconnect'
  | 'metrics_update'
  | 'security_alert'
  | 'task_update'
  | 'agent_status_change'
  | 'error';

// Callback types for different events
type MetricsCallback = (metrics: SystemMetrics) => void;
type SecurityCallback = (event: SecurityEvent) => void;
type TaskCallback = (task: ActiveTask) => void;
type AgentStatusCallback = (agentName: string, status: string) => void;
type ErrorCallback = (error: Error) => void;
type ConnectionCallback = () => void;

// WebSocket service for real-time updates
class WebSocketService {
  private socket: Socket | null = null;
  private wsUrl: string;
  private callbacks: Map<WebSocketEvent, Function[]>;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;
  private reconnectInterval: number = 3000; // 3 seconds
  
  constructor(url: string = process.env.REACT_APP_WS_URL || 'ws://localhost:8000') {
    this.wsUrl = url;
    this.callbacks = new Map();
  }
  
  // Initialize and connect to the WebSocket server
  public connect(token?: string): void {
    if (this.socket) {
      return; // Already connected
    }
    
    // Configure socket options
    const options = {
      transports: ['websocket'],
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: this.reconnectInterval,
      auth: token ? { token } : undefined
    };
    
    // Create socket instance
    this.socket = io(this.wsUrl, options);
    
    // Set up event listeners
    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.triggerCallbacks('connect');
    });
    
    this.socket.on('disconnect', () => {
      console.log('WebSocket disconnected');
      this.triggerCallbacks('disconnect');
    });
    
    this.socket.on('metrics_update', (metrics: SystemMetrics) => {
      this.triggerCallbacks('metrics_update', metrics);
    });
    
    this.socket.on('security_alert', (event: SecurityEvent) => {
      this.triggerCallbacks('security_alert', event);
    });
    
    this.socket.on('task_update', (task: ActiveTask) => {
      this.triggerCallbacks('task_update', task);
    });
    
    this.socket.on('agent_status_change', (agentName: string, status: string) => {
      this.triggerCallbacks('agent_status_change', agentName, status);
    });
    
    this.socket.on('error', (error: Error) => {
      console.error('WebSocket error:', error);
      this.triggerCallbacks('error', error);
    });
    
    this.socket.on('connect_error', (error: Error) => {
      console.error('WebSocket connection error:', error);
      this.reconnectAttempts++;
      
      if (this.reconnectAttempts > this.maxReconnectAttempts) {
        console.error('Max reconnection attempts reached');
        this.triggerCallbacks('error', new Error('Failed to connect to real-time updates'));
      }
    });
  }
  
  // Disconnect from the WebSocket server
  public disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }
  
  // Subscribe to WebSocket events
  public on(event: 'connect', callback: ConnectionCallback): void;
  public on(event: 'disconnect', callback: ConnectionCallback): void;
  public on(event: 'metrics_update', callback: MetricsCallback): void;
  public on(event: 'security_alert', callback: SecurityCallback): void;
  public on(event: 'task_update', callback: TaskCallback): void;
  public on(event: 'agent_status_change', callback: AgentStatusCallback): void;
  public on(event: 'error', callback: ErrorCallback): void;
  public on(event: WebSocketEvent, callback: Function): void {
    if (!this.callbacks.has(event)) {
      this.callbacks.set(event, []);
    }
    
    this.callbacks.get(event)?.push(callback);
  }
  
  // Unsubscribe from WebSocket events
  public off(event: WebSocketEvent, callback?: Function): void {
    if (!callback) {
      // Remove all callbacks for this event
      this.callbacks.delete(event);
      return;
    }
    
    // Remove specific callback
    const callbacks = this.callbacks.get(event);
    if (callbacks) {
      const index = callbacks.indexOf(callback);
      if (index !== -1) {
        callbacks.splice(index, 1);
      }
    }
  }
  
  // Trigger callbacks for a specific event
  private triggerCallbacks(event: WebSocketEvent, ...args: any[]): void {
    const callbacks = this.callbacks.get(event);
    if (callbacks) {
      callbacks.forEach(callback => {
        try {
          callback(...args);
        } catch (error) {
          console.error(`Error in ${event} callback:`, error);
        }
      });
    }
  }
  
  // Check if WebSocket is connected
  public isConnected(): boolean {
    return this.socket?.connected || false;
  }
}

// Create and export a singleton instance
const websocketService = new WebSocketService();
export default websocketService;
