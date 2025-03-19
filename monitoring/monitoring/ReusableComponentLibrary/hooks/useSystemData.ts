// monitoring/dashboard/hooks/useSystemData.ts
import { useState, useEffect, useCallback } from 'react';
import { SystemData, SystemMetrics } from '../types/system';
import { fetchSystemData } from '../utils/api';
import useRefreshInterval from './useRefreshInterval';

interface UseSystemDataOptions {
  refreshInterval?: number; // in milliseconds
  autoRefresh?: boolean;
  initialFetch?: boolean;
}

/**
 * Hook for managing system data
 */
const useSystemData = (options: UseSystemDataOptions = {}) => {
  const {
    refreshInterval = 30000, // Default to 30 seconds
    autoRefresh = true,
    initialFetch = true
  } = options;

  const [systemData, setSystemData] = useState<SystemData | null>(null);
  const [loading, setLoading] = useState(initialFetch);
  const [error, setError] = useState<Error | null>(null);

  // Fetch function that will be passed to useRefreshInterval
  const fetchData = useCallback(async () => {
    try {
      const data = await fetchSystemData();
      setSystemData(data.systemData);
      return data.systemData;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to fetch system data');
      setError(error);
      throw error;
    }
  }, []);

  // Use the refresh interval hook
  const {
    isRefreshing,
    lastRefreshed,
    refresh,
    startRefresh,
    stopRefresh,
    isActive
  } = useRefreshInterval(fetchData, {
    intervalMs: refreshInterval,
    autostart: autoRefresh,
    immediate: initialFetch,
    onError: setError
  });

  // On initial mount, handle loading state
  useEffect(() => {
    if (initialFetch) {
      setLoading(true);
      
      fetchData()
        .catch(err => {
          console.error('Error fetching initial system data:', err);
        })
        .finally(() => {
          setLoading(false);
        });
    }
  }, [initialFetch, fetchData]);

  // Helper to get system metrics for a specific component
  const getMetric = (metricName: keyof SystemMetrics): number | null => {
    if (!systemData?.metrics) return null;
    return systemData.metrics[metricName];
  };

  // Helper to check if a system component is online
  const isComponentOnline = (componentName: string): boolean => {
    if (!systemData?.components) return false;
    const component = systemData.components.find(c => c.name === componentName);
    return component?.status === 'online';
  };

  // Return all the data and utilities
  return {
    systemData,
    loading: loading || isRefreshing,
    error,
    lastRefreshed,
    refresh,
    startAutoRefresh: startRefresh,
    stopAutoRefresh: stopRefresh,
    isAutoRefreshActive: isActive,
    
    // Helper functions
    getMetric,
    isComponentOnline
  };
};

export default useSystemData;
