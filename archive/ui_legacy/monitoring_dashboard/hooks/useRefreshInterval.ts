// monitoring/dashboard/hooks/useRefreshInterval.ts
import { useState, useEffect, useCallback, useRef } from 'react';

interface UseRefreshIntervalOptions {
  intervalMs?: number;
  autostart?: boolean;
  immediate?: boolean;
  onError?: (error: any) => void;
}

/**
 * Custom hook to handle periodic data refreshing
 * @param fetchFunction Function to call on each refresh interval
 * @param options Refresh options
 * @returns Object containing refresh state and controls
 */
const useRefreshInterval = <T>(
  fetchFunction: () => Promise<T>,
  options: UseRefreshIntervalOptions = {}
) => {
  const {
    intervalMs = 30000, // Default to 30 seconds
    autostart = true,
    immediate = true,
    onError
  } = options;

  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [data, setData] = useState<T | null>(null);
  const [isActive, setIsActive] = useState(autostart);
  
  // Store interval ID in ref to avoid dependency issues in useEffect
  const intervalIdRef = useRef<NodeJS.Timeout | null>(null);

  // The refresh function
  const refresh = useCallback(async () => {
    setIsRefreshing(true);
    setError(null);
    
    try {
      const result = await fetchFunction();
      setData(result);
      setLastRefreshed(new Date());
      return result;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('An unknown error occurred');
      setError(error);
      if (onError) {
        onError(error);
      }
      throw error;
    } finally {
      setIsRefreshing(false);
    }
  }, [fetchFunction, onError]);

  // Start refresh interval
  const startRefresh = useCallback(() => {
    setIsActive(true);
  }, []);

  // Stop refresh interval
  const stopRefresh = useCallback(() => {
    setIsActive(false);
  }, []);

  // Set up and clean up interval
  useEffect(() => {
    // Initial fetch if immediate is true
    if (immediate && isActive) {
      refresh();
    }
    
    // Only set up interval if active
    if (isActive) {
      intervalIdRef.current = setInterval(refresh, intervalMs);
      
      // Clean up on unmount or when dependencies change
      return () => {
        if (intervalIdRef.current) {
          clearInterval(intervalIdRef.current);
          intervalIdRef.current = null;
        }
      };
    }
    
    return undefined;
  }, [refresh, intervalMs, immediate, isActive]);

  return {
    isRefreshing,
    lastRefreshed,
    error,
    data,
    refresh,
    startRefresh,
    stopRefresh,
    isActive
  };
};

export default useRefreshInterval;
