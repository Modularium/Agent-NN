// src/services/api.ts
import { QueryClient } from '@tanstack/react-query'
import { useAppStore } from '@/store/useAppStore'

// Types
export interface ApiResponse<T = any> {
  data: T
  success: boolean
  message?: string
  errors?: string[]
  meta?: {
    total?: number
    page?: number
    limit?: number
  }
}

export interface ApiError {
  status: number
  message: string
  code?: string
  details?: any
}

class ApiClient {
  private baseURL: string
  private timeout: number
  private retries: number

  constructor() {
    const settings = useAppStore.getState().settings
    this.baseURL = settings.api.baseUrl
    this.timeout = settings.api.timeout
    this.retries = settings.api.retries
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const { setError, setLoading } = useAppStore.getState()
    
    try {
      setLoading(true)
      
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), this.timeout)

      const response = await fetch(`${this.baseURL}${endpoint}`, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new ApiError({
          status: response.status,
          message: errorData.message || `HTTP ${response.status}`,
          code: errorData.code,
          details: errorData
        })
      }

      const data = await response.json()
      setError(null)
      return data

    } catch (error) {
      if (error instanceof ApiError) {
        setError(error.message)
        throw error
      }
      
      if (error.name === 'AbortError') {
        const timeoutError = new ApiError({
          status: 408,
          message: 'Request timeout',
          code: 'TIMEOUT'
        })
        setError(timeoutError.message)
        throw timeoutError
      }

      const networkError = new ApiError({
        status: 0,
        message: 'Network error',
        code: 'NETWORK_ERROR'
      })
      setError(networkError.message)
      throw networkError

    } finally {
      setLoading(false)
    }
  }

  // HTTP Methods
  async get<T>(endpoint: string, params?: Record<string, any>): Promise<ApiResponse<T>> {
    const url = new URL(`${this.baseURL}${endpoint}`)
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          url.searchParams.append(key, String(value))
        }
      })
    }
    
    return this.request<T>(url.pathname + url.search)
  }

  async post<T>(endpoint: string, data?: any): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    })
  }

  async put<T>(endpoint: string, data?: any): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    })
  }

  async patch<T>(endpoint: string, data?: any): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      method: 'PATCH',
      body: data ? JSON.stringify(data) : undefined,
    })
  }

  async delete<T>(endpoint: string): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      method: 'DELETE',
    })
  }

  // File upload
  async upload<T>(endpoint: string, file: File, additionalData?: Record<string, any>): Promise<ApiResponse<T>> {
    const formData = new FormData()
    formData.append('file', file)
    
    if (additionalData) {
      Object.entries(additionalData).forEach(([key, value]) => {
        formData.append(key, String(value))
      })
    }

    return this.request<T>(endpoint, {
      method: 'POST',
      body: formData,
      headers: {
        // Don't set Content-Type for FormData, let browser set it
      },
    })
  }
}

// Singleton instance
export const api = new ApiClient()

// React Query Client
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
    },
    mutations: {
      retry: 1,
    },
  },
})

// Custom ApiError class
export class ApiError extends Error {
  public status: number
  public code?: string
  public details?: any

  constructor({ status, message, code, details }: {
    status: number
    message: string
    code?: string
    details?: any
  }) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.details = details
  }
}

// Utility function for error handling
export const handleApiError = (error: unknown): string => {
  if (error instanceof ApiError) {
    return error.message
  }
  
  if (error instanceof Error) {
    return error.message
  }
  
  return 'An unexpected error occurred'
}
