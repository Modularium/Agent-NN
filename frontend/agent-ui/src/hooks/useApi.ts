// src/hooks/useApi.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api, ApiResponse, handleApiError } from '@/services/api'
import { useAppStore, Agent, Task, User } from '@/store/useAppStore'

// Query Keys
export const queryKeys = {
  agents: ['agents'] as const,
  agent: (id: string) => ['agents', id] as const,
  tasks: ['tasks'] as const,
  task: (id: string) => ['tasks', id] as const,
  user: ['user'] as const,
  metrics: ['metrics'] as const,
  system: ['system'] as const,
  chat: (sessionId: string) => ['chat', sessionId] as const,
}

// Agents API Hooks
export const useAgents = (filters?: {
  status?: string
  domain?: string
  search?: string
}) => {
  return useQuery({
    queryKey: [...queryKeys.agents, filters],
    queryFn: async () => {
      const response = await api.get<Agent[]>('/agents', filters)
      return response.data
    },
    onSuccess: (data) => {
      useAppStore.getState().setAgents(data)
    },
    onError: (error) => {
      useAppStore.getState().addNotification({
        type: 'error',
        title: 'Failed to load agents',
        message: handleApiError(error)
      })
    }
  })
}

export const useAgent = (id: string) => {
  return useQuery({
    queryKey: queryKeys.agent(id),
    queryFn: async () => {
      const response = await api.get<Agent>(`/agents/${id}`)
      return response.data
    },
    enabled: !!id,
    onError: (error) => {
      useAppStore.getState().addNotification({
        type: 'error',
        title: 'Failed to load agent',
        message: handleApiError(error)
      })
    }
  })
}

export const useCreateAgent = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async (agentData: Omit<Agent, 'id'>) => {
      const response = await api.post<Agent>('/agents', agentData)
      return response.data
    },
    onSuccess: (newAgent) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.agents })
      useAppStore.getState().addAgent(newAgent)
      useAppStore.getState().addNotification({
        type: 'success',
        title: 'Agent created',
        message: `${newAgent.name} has been created successfully`
      })
    },
    onError: (error) => {
      useAppStore.getState().addNotification({
        type: 'error',
        title: 'Failed to create agent',
        message: handleApiError(error)
      })
    }
  })
}

export const useUpdateAgent = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async ({ id, updates }: { id: string; updates: Partial<Agent> }) => {
      const response = await api.patch<Agent>(`/agents/${id}`, updates)
      return response.data
    },
    onSuccess: (updatedAgent) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.agents })
      queryClient.invalidateQueries({ queryKey: queryKeys.agent(updatedAgent.id) })
      useAppStore.getState().updateAgent(updatedAgent.id, updatedAgent)
      useAppStore.getState().addNotification({
        type: 'success',
        title: 'Agent updated',
        message: `${updatedAgent.name} has been updated successfully`
      })
    },
    onError: (error) => {
      useAppStore.getState().addNotification({
        type: 'error',
        title: 'Failed to update agent',
        message: handleApiError(error)
      })
    }
  })
}

export const useDeleteAgent = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async (id: string) => {
      await api.delete(`/agents/${id}`)
      return id
    },
    onSuccess: (deletedId) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.agents })
      useAppStore.getState().removeAgent(deletedId)
      useAppStore.getState().addNotification({
        type: 'success',
        title: 'Agent deleted',
        message: 'Agent has been deleted successfully'
      })
    },
    onError: (error) => {
      useAppStore.getState().addNotification({
        type: 'error',
        title: 'Failed to delete agent',
        message: handleApiError(error)
      })
    }
  })
}

// Tasks API Hooks
export const useTasks = (filters?: {
  status?: string
  priority?: string
  agentId?: string
  search?: string
  limit?: number
  offset?: number
}) => {
  return useQuery({
    queryKey: [...queryKeys.tasks, filters],
    queryFn: async () => {
      const response = await api.get<Task[]>('/tasks', filters)
      return response.data
    },
    onSuccess: (data) => {
      useAppStore.getState().setTasks(data)
    },
    onError: (error) => {
      useAppStore.getState().addNotification({
        type: 'error',
        title: 'Failed to load tasks',
        message: handleApiError(error)
      })
    }
  })
}

export const useTask = (id: string) => {
  return useQuery({
    queryKey: queryKeys.task(id),
    queryFn: async () => {
      const response = await api.get<Task>(`/tasks/${id}`)
      return response.data
    },
    enabled: !!id,
    onError: (error) => {
      useAppStore.getState().addNotification({
        type: 'error',
        title: 'Failed to load task',
        message: handleApiError(error)
      })
    }
  })
}

export const useCreateTask = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async (taskData: Omit<Task, 'id'>) => {
      const response = await api.post<Task>('/tasks', taskData)
      return response.data
    },
    onSuccess: (newTask) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.tasks })
      useAppStore.getState().addTask(newTask)
      useAppStore.getState().addNotification({
        type: 'success',
        title: 'Task created',
        message: `${newTask.title} has been created successfully`
      })
    },
    onError: (error) => {
      useAppStore.getState().addNotification({
        type: 'error',
        title: 'Failed to create task',
        message: handleApiError(error)
      })
    }
  })
}

export const useUpdateTask = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async ({ id, updates }: { id: string; updates: Partial<Task> }) => {
      const response = await api.patch<Task>(`/tasks/${id}`, updates)
      return response.data
    },
    onSuccess: (updatedTask) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.tasks })
      queryClient.invalidateQueries({ queryKey: queryKeys.task(updatedTask.id) })
      useAppStore.getState().updateTask(updatedTask.id, updatedTask)
    },
    onError: (error) => {
      useAppStore.getState().addNotification({
        type: 'error',
        title: 'Failed to update task',
        message: handleApiError(error)
      })
    }
  })
}

// Chat API Hooks
export const useChatSession = (sessionId: string) => {
  return useQuery({
    queryKey: queryKeys.chat(sessionId),
    queryFn: async () => {
      const response = await api.get(`/chat/sessions/${sessionId}`)
      return response.data
    },
    enabled: !!sessionId,
    refetchInterval: 5000, // Poll every 5 seconds for real-time updates
  })
}

export const useSendMessage = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async (messageData: {
      sessionId: string
      content: string
      taskType?: string
      metadata?: Record<string, any>
    }) => {
      const response = await api.post(`/chat/sessions/${messageData.sessionId}/messages`, messageData)
      return response.data
    },
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.chat(variables.sessionId) })
    },
    onError: (error) => {
      useAppStore.getState().addNotification({
        type: 'error',
        title: 'Failed to send message',
        message: handleApiError(error)
      })
    }
  })
}

// System Metrics Hooks
export const useSystemMetrics = () => {
  return useQuery({
    queryKey: queryKeys.metrics,
    queryFn: async () => {
      const response = await api.get('/metrics/system')
      return response.data
    },
    refetchInterval: 30000, // Refresh every 30 seconds
    onError: (error) => {
      useAppStore.getState().addNotification({
        type: 'error',
        title: 'Failed to load system metrics',
        message: handleApiError(error)
      })
    }
  })
}

export const useSystemHealth = () => {
  return useQuery({
    queryKey: [...queryKeys.system, 'health'],
    queryFn: async () => {
      const response = await api.get('/system/health')
      return response.data
    },
    refetchInterval: 10000, // Refresh every 10 seconds
    retry: 1, // Don't retry too aggressively for health checks
  })
}

// User Management Hooks
export const useCurrentUser = () => {
  return useQuery({
    queryKey: queryKeys.user,
    queryFn: async () => {
      const response = await api.get<User>('/user/me')
      return response.data
    },
    onSuccess: (user) => {
      useAppStore.getState().setUser(user)
    },
    onError: (_error: unknown) => {
      useAppStore.getState().setAuthenticated(false)
    }
  })
}

export const useLogin = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async (credentials: { email: string; password: string }) => {
      const response = await api.post<{ user: User; token: string }>('/auth/login', credentials)
      return response.data
    },
    onSuccess: (data) => {
      localStorage.setItem('auth_token', data.token)
      useAppStore.getState().setUser(data.user)
      queryClient.invalidateQueries({ queryKey: queryKeys.user })
      useAppStore.getState().addNotification({
        type: 'success',
        title: 'Welcome back!',
        message: 'You have been logged in successfully'
      })
    },
    onError: (error) => {
      useAppStore.getState().addNotification({
        type: 'error',
        title: 'Login failed',
        message: handleApiError(error)
      })
    }
  })
}

export const useLogout = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async () => {
      await api.post('/auth/logout')
    },
    onSuccess: () => {
      localStorage.removeItem('auth_token')
      useAppStore.getState().logout()
      queryClient.clear()
      useAppStore.getState().addNotification({
        type: 'info',
        title: 'Logged out',
        message: 'You have been logged out successfully'
      })
    },
    onError: (error) => {
      // Still log out locally even if server request fails
      localStorage.removeItem('auth_token')
      useAppStore.getState().logout()
      queryClient.clear()
    }
  })
}

// File Upload Hook
export const useFileUpload = () => {
  return useMutation({
    mutationFn: async ({ file, endpoint, additionalData }: {
      file: File
      endpoint: string
      additionalData?: Record<string, any>
    }) => {
      const response = await api.upload(endpoint, file, additionalData)
      return response.data
    },
    onSuccess: () => {
      useAppStore.getState().addNotification({
        type: 'success',
        title: 'Upload successful',
        message: 'File has been uploaded successfully'
      })
    },
    onError: (error) => {
      useAppStore.getState().addNotification({
        type: 'error',
        title: 'Upload failed',
        message: handleApiError(error)
      })
    }
  })
}
