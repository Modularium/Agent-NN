// src/hooks/useWebSocket.ts
import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';
import websocketService from '../utils/websocketService';
import { SystemMetrics, SecurityEvent, ActiveTask } from '../types/system';
import { useNotification } from '../components/common/NotificationSystem';

interface UseWebSocketOptions {
  autoConnect?: boolean;
  onMetricsUpdate?: (metrics: SystemMetrics) => void;
  onSecurityAlert?: (event: SecurityEvent) => void;
  onTaskUpdate?: (task: ActiveTask) => void;
  onAgentStatusChange?: (agentName: string, status: string) => void;
  onError?: (error: Error) => void;
  showNotifications?: boolean;
}

/**
 * Custom hook for using WebSocket real-time updates
 */
const useWebSocket = (options: UseWebSocketOptions = {}) => {
  const {
    autoConnect = true,
    onMetricsUpdate,
    onSecurityAlert,
    onTaskUpdate,
    onAgentStatusChange,
    onError,
    showNotifications = true
  } = options;
  
  const { user } = useAuth();
  const { addNotification } = useNotification();
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<Error | null>(null);
  
  // Connect to WebSocket
  const connect = useCallback(() => {
    const token = localStorage.getItem('authToken');
    websocketService.connect(token || undefined);
  }, []);
  
  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    websocketService.disconnect();
  }, []);
  
  // Set up event listeners and connection
  useEffect(() => {
    // Connection events
    const handleConnect = () => {
      setIsConnected(true);
      setConnectionError(null);
    };
    
    const handleDisconnect = () => {
      setIsConnected(false);
    };
    
    // Data events
    const handleMetricsUpdate = (metrics: SystemMetrics) => {
      if (onMetricsUpdate) {
        onMetricsUpdate(metrics);
      }
    };
    
    const handleSecurityAlert = (event: SecurityEvent) => {
      if (onSecurityAlert) {
        onSecurityAlert(event);
      }
      
      if (showNotifications) {
        // Show notification for security alerts
        addNotification({
          type: event.severity === 'high' ? 'error' : 
                event.severity === 'medium' ? 'warning' : 'info',
          title: 'Security Alert',
          message: event.details,
          duration: event.severity === 'high' ? 0 : 8000, // Don't auto-dismiss high severity alerts
        });
      }
    };
    
    const handleTaskUpdate = (task: ActiveTask) => {
      if (onTaskUpdate) {
        onTaskUpdate(task);
      }
      
      if (showNotifications && task.status === 'completed') {
        // Show notification for completed tasks
        addNotification({
          type: 'success',
          title: 'Task Completed',
          message: `Task ${task.id} has been completed by ${task.agent}`,
          duration: 5000,
        });
      }
    };
    
    const handleAgentStatusChange = (agentName: string, status: string) => {
      if (onAgentStatusChange) {
        onAgentStatusChange(agentName, status);
      }
      
      if (showNotifications && status === 'error') {
        // Show notification for agent errors
        addNotification({
          type: 'error',
          title: 'Agent Error',
          message: `Agent "${agentName}" has encountered an error`,
          duration: 8000,
        });
      }
    };
    
    const handleError = (error: Error) => {
      setConnectionError(error);
      
      if (onError) {
        onError(error);
      }
      
      if (showNotifications) {
        addNotification({
          type: 'error',
          title: 'WebSocket Error',
          message: error.message,
          duration: 8000,
        });
      }
    };
    
    // Register event handlers
    websocketService.on('connect', handleConnect);
    websocketService.on('disconnect', handleDisconnect);
    websocketService.on('metrics_update', handleMetricsUpdate);
    websocketService.on('security_alert', handleSecurityAlert);
    websocketService.on('task_update', handleTaskUpdate);
    websocketService.on('agent_status_change', handleAgentStatusChange);
    websocketService.on('error', handleError);
    
    // Connect if autoConnect is true and user is authenticated
    if (autoConnect && user) {
      connect();
    }
    
    // Cleanup function
    return () => {
      websocketService.off('connect', handleConnect);
      websocketService.off('disconnect', handleDisconnect);
      websocketService.off('metrics_update', handleMetricsUpdate);
      websocketService.off('security_alert', handleSecurityAlert);
      websocketService.off('task_update', handleTaskUpdate);
      websocketService.off('agent_status_change', handleAgentStatusChange);
      websocketService.off('error', handleError);
    };
  }, [
    autoConnect,
    user,
    onMetricsUpdate,
    onSecurityAlert,
    onTaskUpdate,
    onAgentStatusChange,
    onError,
    showNotifications,
    addNotification,
    connect
  ]);
  
  return {
    isConnected,
    connectionError,
    connect,
    disconnect
  };
};

export default useWebSocket;
