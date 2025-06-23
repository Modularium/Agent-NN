// monitoring/dashboard/hooks/useApiService.ts
import { useState, useCallback, useRef } from 'react';
import { useAuth } from '../context/AuthContext';
import { useNotification } from '../components/common/NotificationSystem';

interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  body?: any;
  headers?: Record<string, string>;
  showSuccessNotification?: boolean;
  showErrorNotification?: boolean;
  notificationTitle?: string;
  requireAuth?: boolean;
  timeout?: number;
}

interface ApiServiceHook {
  request: <T>(endpoint: string, options?: RequestOptions) => Promise<T>;
  get: <T>(endpoint: string, options?: Omit<RequestOptions, 'method' | 'body'>) => Promise<T>;
  post: <T>(endpoint: string, data?: any, options?: Omit<RequestOptions, 'method' | 'body'>) => Promise<T>;
  put: <T>(endpoint: string, data?: any, options?: Omit<RequestOptions, 'method' | 'body'>) => Promise<T>;
  delete: <T>(endpoint: string, options?: Omit<RequestOptions, 'method' | 'body'>) => Promise<T>;
  patch: <T>(endpoint: string, data?: any, options?: Omit<RequestOptions, 'method' | 'body'>) => Promise<T>;
  loading: boolean;
  error: Error | null;
  clearError: () => void;
  abortRequest: () => void;
}

/**
 * Hook providing API service functionality with authentication, error handling, and notifications
 */
const useApiService = (): ApiServiceHook => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const { user } = useAuth();
  const { addNotification } = useNotification();
  const abortControllerRef = useRef<AbortController | null>(null);

  // API base URL from environment variables or default
  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

  // Clear any existing errors
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // Abort any ongoing requests
  const abortRequest = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  }, []);

  // Main request function
  const request = useCallback(async <T>(
    endpoint: string, 
    options: RequestOptions = {}
  ): Promise<T> => {
    const {
      method = 'GET',
      body,
      headers = {},
      showSuccessNotification = false,
      showErrorNotification = true,
      notificationTitle,
      requireAuth = true,
      timeout = 30000 // Default 30s timeout
    } = options;

    // Create new AbortController
    abortRequest();
    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    // Set timeout
    const timeoutId = setTimeout(() => {
      abortRequest();
      throw new Error('Request timeout');
    }, timeout);

    try {
      setLoading(true);
      clearError();

      // Format URL
      const url = endpoint.startsWith('http') ? endpoint : `${API_BASE_URL}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;

      // Set up headers
      const requestHeaders: Record<string, string> = {
        'Content-Type': 'application/json',
        ...headers
      };

      // Add auth token if required and available
      if (requireAuth) {
        const token = localStorage.getItem('authToken');
        if (token) {
          requestHeaders['Authorization'] = `Bearer ${token}`;
        } else if (requireAuth && !user) {
          throw new Error('Authentication required');
        }
      }

      // Prepare body for non-GET requests
      const requestBody = body ? JSON.stringify(body) : undefined;

      // Make request
      const response = await fetch(url, {
        method,
        headers: requestHeaders,
        body: requestBody,
        signal
      });

      // Clear timeout
      clearTimeout(timeoutId);

      // Handle non-2xx responses
      if (!response.ok) {
        let errorData;
        try {
          errorData = await response.json();
        } catch {
          errorData = { message: 'Unknown error' };
        }

        const errorMessage = errorData.message || `Request failed with status ${response.status}`;
        throw new Error(errorMessage);
      }

      // Parse JSON response (or return null for 204 No Content)
      const data = response.status === 204 ? null : await response.json();

      // Show success notification if requested
      if (showSuccessNotification) {
        addNotification({
          type: 'success',
          title: notificationTitle || 'Success',
          message: 'Operation completed successfully',
          duration: 5000
        });
      }

      return data as T;
    } catch (err) {
      // Handle errors
      const error = err instanceof Error ? err : new Error('An unknown error occurred');
      
      // Only set error state if not aborted
      if (error.name !== 'AbortError') {
        setError(error);
        
        // Show error notification if requested
        if (showErrorNotification) {
          addNotification({
            type: 'error',
            title: notificationTitle || 'Error',
            message: error.message,
            duration: 8000
          });
        }
      }
      
      throw error;
    } finally {
      clearTimeout(timeoutId);
      if (abortControllerRef.current?.signal === signal) {
        abortControllerRef.current = null;
      }
      setLoading(false);
    }
  }, [API_BASE_URL, addNotification, abortRequest, clearError, user]);

  // Convenience wrappers for different HTTP methods
  const get = useCallback(<T>(endpoint: string, options?: Omit<RequestOptions, 'method' | 'body'>) => {
    return request<T>(endpoint, { ...options, method: 'GET' });
  }, [request]);

  const post = useCallback(<T>(endpoint: string, data?: any, options?: Omit<RequestOptions, 'method' | 'body'>) => {
    return request<T>(endpoint, { ...options, method: 'POST', body: data });
  }, [request]);

  const put = useCallback(<T>(endpoint: string, data?: any, options?: Omit<RequestOptions, 'method' | 'body'>) => {
    return request<T>(endpoint, { ...options, method: 'PUT', body: data });
  }, [request]);

  const del = useCallback(<T>(endpoint: string, options?: Omit<RequestOptions, 'method' | 'body'>) => {
    return request<T>(endpoint, { ...options, method: 'DELETE' });
  }, [request]);

  const patch = useCallback(<T>(endpoint: string, data?: any, options?: Omit<RequestOptions, 'method' | 'body'>) => {
    return request<T>(endpoint, { ...options, method: 'PATCH', body: data });
  }, [request]);

  return {
    request,
    get,
    post,
    put,
    delete: del,
    patch,
    loading,
    error,
    clearError,
    abortRequest
  };
};

export default useApiService;
