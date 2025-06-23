import { Task, Agent, Message } from '../types';

// API base URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_PREFIX = '/smolitux';

// WebSocket URL
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
const WS_PREFIX = '/smolitux/ws';

// API client
class ApiClient {
  private baseUrl: string;
  private wsUrl: string;
  private socket: WebSocket | null = null;
  private messageHandlers: ((message: any) => void)[] = [];

  constructor() {
    this.baseUrl = `${API_BASE_URL}${API_PREFIX}`;
    this.wsUrl = `${WS_URL}${WS_PREFIX}`;
  }

  // REST API methods
  async getTasks(): Promise<Task[]> {
    const response = await fetch(`${this.baseUrl}/tasks`);
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    return response.json();
  }

  async getTask(taskId: string): Promise<Task> {
    const response = await fetch(`${this.baseUrl}/tasks/${taskId}`);
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    return response.json();
  }

  async createTask(taskDescription: string, context?: string): Promise<any> {
    const response = await fetch(`${this.baseUrl}/tasks`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        description: taskDescription,
        context
      })
    });
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    return response.json();
  }

  async getAgents(): Promise<Agent[]> {
    const response = await fetch(`${this.baseUrl}/agents`);
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    return response.json();
  }

  async sendChatMessage(message: string, agent: string, sessionId?: string): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, agent, session_id: sessionId })
    })
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`)
    }
    return response.json()
  }

  async getChatHistory(sessionId: string): Promise<Message[]> {
    const response = await fetch(`${API_BASE_URL}/chat/history/${sessionId}`)
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`)
    }
    return response.json()
  }

  async sendFeedback(sessionId: string, index: number, rating: string, comment?: string): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/chat/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, index, rating, comment })
    })
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`)
    }
    return response.json()
  }

  // WebSocket methods
  connectWebSocket(): WebSocket {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      return this.socket;
    }

    this.socket = new WebSocket(this.wsUrl);

    this.socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.messageHandlers.forEach(handler => handler(data));
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    this.socket.onclose = () => {
      console.log('WebSocket connection closed');
      this.socket = null;
    };

    this.socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.socket?.close();
      this.socket = null;
    };

    return this.socket;
  }

  disconnectWebSocket(): void {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }

  sendMessage(message: string, context?: string): void {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      this.connectWebSocket();
    }

    this.socket?.send(JSON.stringify({
      task_description: message,
      context
    }));
  }

  onMessage(handler: (message: any) => void): () => void {
    this.messageHandlers.push(handler);
    return () => {
      this.messageHandlers = this.messageHandlers.filter(h => h !== handler);
    };
  }
}

// Create and export a singleton instance
const apiClient = new ApiClient();
export default apiClient;
