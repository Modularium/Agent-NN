// monitoring/dashboard/hooks/useMetrics.ts
import { useState, useEffect, useMemo } from 'react';
import { TimeSeriesData } from '../types/metrics';

interface UseMetricsOptions {
  timeRange?: '1h' | '6h' | '24h' | '7d' | '30d' | 'all';
  threshold?: {
    warning?: number;
    critical?: number;
  };
}

/**
 * Hook for processing and analyzing time series metrics data
 */
const useMetrics = (data: TimeSeriesData[], options: UseMetricsOptions = {}) => {
  const {
    timeRange = '24h',
    threshold
  } = options;

  const [filteredData, setFilteredData] = useState<TimeSeriesData[]>([]);

  // Filter data based on time range
  useEffect(() => {
    if (timeRange === 'all' || !data || data.length === 0) {
      setFilteredData(data);
      return;
    }

    const now = new Date();
    let cutoff = new Date();

    switch (timeRange) {
      case '1h':
        cutoff.setHours(now.getHours() - 1);
        break;
      case '6h':
        cutoff.setHours(now.getHours() - 6);
        break;
      case '24h':
        cutoff.setHours(now.getHours() - 24);
        break;
      case '7d':
        cutoff.setDate(now.getDate() - 7);
        break;
      case '30d':
        cutoff.setDate(now.getDate() - 30);
        break;
    }

    const filtered = data.filter(item => new Date(item.timestamp) >= cutoff);
    setFilteredData(filtered);
  }, [data, timeRange]);

  // Calculate statistics from the filtered data
  const stats = useMemo(() => {
    if (!filteredData || filteredData.length === 0) {
      return {
        current: null,
        min: null,
        max: null,
        avg: null,
        median: null,
        trend: null,
        status: 'normal' as 'normal' | 'warning' | 'critical'
      };
    }

    // Sort data by timestamp to ensure correct calculations
    const sortedData = [...filteredData].sort(
      (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    // Get values array
    const values = sortedData.map(item => item.value);

    // Calculate statistics
    const current = values[values.length - 1];
    const min = Math.min(...values);
    const max = Math.max(...values);
    const avg = values.reduce((sum, val) => sum + val, 0) / values.length;

    // Calculate median
    const sortedValues = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sortedValues.length / 2);
    const median = sortedValues.length % 2 === 0
      ? (sortedValues[mid - 1] + sortedValues[mid]) / 2
      : sortedValues[mid];

    // Calculate trend (percentage change over the time period)
    // Using first and last values
    const first = values[0];
    const last = values[values.length - 1];
    const trend = first !== 0 ? ((last - first) / first) * 100 : 0;

    // Determine status based on thresholds
    let status: 'normal' | 'warning' | 'critical' = 'normal';
    
    if (threshold) {
      if (threshold.critical !== undefined && current >= threshold.critical) {
        status = 'critical';
      } else if (threshold.warning !== undefined && current >= threshold.warning) {
        status = 'warning';
      }
    }

    return {
      current,
      min,
      max,
      avg,
      median,
      trend,
      status
    };
  }, [filteredData, threshold]);

  // Generate chart data
  const chartData = useMemo(() => {
    if (!filteredData || filteredData.length === 0) {
      return {
        labels: [],
        values: [],
        timestamps: []
      };
    }

    // Sort by timestamp
    const sortedData = [...filteredData].sort(
      (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    // Format labels based on time range
    const labels = sortedData.map(item => {
      const date = new Date(item.timestamp);
      if (timeRange === '1h' || timeRange === '6h') {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      } else if (timeRange === '24h') {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      } else {
        return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
      }
    });

    return {
      labels,
      values: sortedData.map(item => item.value),
      timestamps: sortedData.map(item => item.timestamp)
    };
  }, [filteredData, timeRange]);

  // Helper function to simplify data for smoother charts
  const getSimplifiedData = (maxPoints = 50) => {
    if (chartData.values.length <= maxPoints) {
      return chartData;
    }

    const skip = Math.ceil(chartData.values.length / maxPoints);
    const simplified = {
      labels: [] as string[],
      values: [] as number[],
      timestamps: [] as string[]
    };

    for (let i = 0; i < chartData.values.length; i += skip) {
      simplified.labels.push(chartData.labels[i]);
      simplified.values.push(chartData.values[i]);
      simplified.timestamps.push(chartData.timestamps[i]);
    }

    // Always include the most recent point
    const lastIndex = chartData.values.length - 1;
    if ((chartData.values.length - 1) % skip !== 0) {
      simplified.labels.push(chartData.labels[lastIndex]);
      simplified.values.push(chartData.values[lastIndex]);
      simplified.timestamps.push(chartData.timestamps[lastIndex]);
    }

    return simplified;
  };

  // Anomaly detection (simple implementation using z-score)
  const detectAnomalies = (threshold = 2.5) => {
    if (!filteredData || filteredData.length === 0) {
      return [];
    }

    const values = filteredData.map(item => item.value);
    const mean = stats.avg || 0;
    
    // Calculate standard deviation
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);

    // Find anomalies (points where z-score > threshold)
    return filteredData.filter(item => {
      const zScore = Math.abs((item.value - mean) / (stdDev || 1)); // Avoid division by zero
      return zScore > threshold;
    });
  };

  return {
    // Raw and filtered data
    rawData: data,
    filteredData,
    
    // Statistics
    stats,
    
    // Chart data
    chartData,
    getSimplifiedData,
    
    // Analysis functions
    detectAnomalies
  };
};

export default useMetrics;
