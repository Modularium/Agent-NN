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
type MetricsCallback = (metrics: SystemMet
