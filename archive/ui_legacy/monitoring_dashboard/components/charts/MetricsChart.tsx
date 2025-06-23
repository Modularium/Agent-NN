// MetricsChart.tsx - Component for displaying system metric charts
import React, { useState, useEffect } from 'react';
import { ChevronDown, Maximize, Download, RefreshCw, CalendarDays } from 'lucide-react';
import { useTheme } from '../../context/ThemeContext';
import { TimeSeriesData } from '../../types/metrics';

interface MetricsChartProps {
  title: string;
  metricName: string;
  data: TimeSeriesData[];
  unit?: string;
  thresholds?: {
    warning?: number;
    critical?: number;
  };
  timeRanges?: { 
    label: string; 
    value: string; 
    duration: number; // in milliseconds
  }[];
  height?: number;
  showControls?: boolean;
  onRefresh?: () => Promise<void>;
  refreshInterval?: number; // in seconds
  className?: string;
}

const MetricsChart: React.FC<MetricsChartProps> = ({
  title,
  metricName,
  data,
  unit = '',
  thresholds,
  timeRanges = [
    { label: '1h', value: '1h', duration: 60 * 60 * 1000 },
    { label: '6h', value: '6h', duration: 6 * 60 * 60 * 1000 },
    { label: '24h', value: '24h', duration: 24 * 60 * 60 * 1000 },
    { label: '7d', value: '7d', duration: 7 * 24 * 60 * 60 * 1000 },
    { label: '30d', value: '30d', duration: 30 * 24 * 60 * 60 * 1000 },
  ],
  height = 300,
  showControls = true,
  onRefresh,
  refreshInterval,
  className = '',
}) => {
  const { themeMode } = useTheme();
  const [selectedTimeRange, setSelectedTimeRange] = useState(timeRanges[2].value); // Default to 24h
  const [showTimeRangeDropdown, setShowTimeRangeDropdown] = useState(false);
  const [isFullScreen, setIsFullScreen] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  
  // Auto-refresh timer
  useEffect(() => {
    if (!refreshInterval || !onRefresh) return;
    
    const timer = setInterval(() => {
      onRefresh();
    }, refreshInterval * 1000);
    
    return () => clearInterval(timer);
  }, [refreshInterval, onRefresh]);
  
  // Apply time range filter
  const getFilteredData = () => {
    const selectedRange = timeRanges.find(range => range.value === selectedTimeRange);
    if (!selectedRange) return data;
    
    const cutoff = new Date().getTime() - selectedRange.duration;
    return data.filter(point => new Date(point.timestamp).getTime() >= cutoff);
  };
  
  const filteredData = getFilteredData();
  
  // Calculate chart dimensions and scaling
  const chartWidth = isFullScreen ? 800 : 600;
  const chartHeight = height;
  const padding = { top: 20, right: 30, bottom: 40, left: 50 };
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;
  
  // Find min and max values for scaling
  const minValue = Math.min(...filteredData.map(d => d.value));
  const maxValue = Math.max(...filteredData.map(d => d.value));
  const valueRange = maxValue - minValue;
  const chartMin = Math.max(0, minValue - valueRange * 0.1);
  const chartMax = maxValue + valueRange * 0.1;
  
  // Scale values to chart coordinates
  const scaleY = (value: number): number => {
    return padding.top + innerHeight - ((value - chartMin) / (chartMax - chartMin)) * innerHeight;
  };
  
  const scaleX = (index: number, total: number): number => {
    return padding.left + (index / (total - 1)) * innerWidth;
  };
  
  // Format time labels based on selected time range
  const formatTimeLabel = (timestamp: string): string => {
    const date = new Date(timestamp);
    const selectedDuration = timeRanges.find(range => range.value === selectedTimeRange)?.duration || 0;
    
    if (selectedDuration <= 6 * 60 * 60 * 1000) { // 6 hours or less
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (selectedDuration <= 24 * 60 * 60 * 1000) { // 24 hours or less
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else {
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
  };
  
  // Refresh data handler
  const handleRefresh = async () => {
    if (!onRefresh || isRefreshing) return;
    
    setIsRefreshing(true);
    try {
      await onRefresh();
    } finally {
      setIsRefreshing(false);
    }
  };
  
  // Download chart as SVG
  const handleDownload = () => {
    const svgElement = document.getElementById(`metrics-chart-${metricName.replace(/\s+/g, '-')}`);
    if (!svgElement) return;
    
    const svgData = new XMLSerializer().serializeToString(svgElement);
    const blob = new Blob([svgData], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `${metricName.replace(/\s+/g, '-')}-${selectedTimeRange}.svg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  // Toggle fullscreen
  const handleToggleFullscreen = () => {
    setIsFullScreen(!isFullScreen);
  };
  
  // Generate path for the line chart
  const generateLinePath = (): string => {
    if (filteredData.length < 2) return '';
    
    return filteredData
      .map((point, index) => {
        const x = scaleX(index, filteredData.length);
        const y = scaleY(point.value);
        return `${index === 0 ? 'M' : 'L'} ${x} ${y}`;
      })
      .join(' ');
  };
  
  // Generate fill area below the line
  const generateAreaPath = (): string => {
    if (filteredData.length < 2) return '';
    
    const linePath = filteredData
      .map((point, index) => {
        const x = scaleX(index, filteredData.length);
        const y = scaleY(point.value);
        return `${index === 0 ? 'M' : 'L'} ${x} ${y}`;
      })
      .join(' ');
    
    const firstX = scaleX(0, filteredData.length);
    const lastX = scaleX(filteredData.length - 1, filteredData.length);
    const baseline = padding.top + innerHeight;
    
    return `${linePath} L ${lastX} ${baseline} L ${firstX} ${baseline} Z`;
  };
  
  // Generate threshold lines
  const renderThresholdLines = () => {
    if (!thresholds) return null;
    
    return (
      <>
        {thresholds.warning && (
          <g>
            <line
              x1={padding.left}
              y1={scaleY(thresholds.warning)}
              x2={padding.left + innerWidth}
              y2={scaleY(thresholds.warning)}
              stroke="#F59E0B"
              strokeWidth="1"
              strokeDasharray="4,4"
            />
            <text
              x={padding.left + innerWidth + 5}
              y={scaleY(thresholds.warning)}
              fontSize="10"
              fill="#F59E0B"
              dominantBaseline="middle"
            >
              Warning
            </text>
          </g>
        )}
        {thresholds.critical && (
          <g>
            <line
              x1={padding.left}
              y1={scaleY(thresholds.critical)}
              x2={padding.left + innerWidth}
              y2={scaleY(thresholds.critical)}
              stroke="#EF4444"
              strokeWidth="1"
              strokeDasharray="4,4"
            />
            <text
              x={padding.left + innerWidth + 5}
              y={scaleY(thresholds.critical)}
              fontSize="10"
              fill="#EF4444"
              dominantBaseline="middle"
            >
              Critical
            </text>
          </g>
        )}
      </>
    );
  };
  
  // Generate time-based x-axis labels
  const renderXAxisLabels = () => {
    // For better readability, we'll show only a subset of labels
    const totalLabels = filteredData.length;
    const labelCount = Math.min(6, totalLabels);
    const step = Math.max(1, Math.floor(totalLabels / labelCount));
    
    return filteredData
      .filter((_, index) => index % step === 0 || index === totalLabels - 1)
      .map((point, index) => {
        const x = scaleX(
          filteredData.findIndex(d => d.timestamp === point.timestamp), 
          filteredData.length
        );
        
        return (
          <g key={`x-label-${index}`} transform={`translate(${x}, ${padding.top + innerHeight})`}>
            <line
              y2="6"
              stroke={themeMode === 'dark' ? '#6B7280' : '#9CA3AF'}
              strokeWidth="1"
            />
            <text
              y="20"
              textAnchor="middle"
              fontSize="10"
              fill={themeMode === 'dark' ? '#9CA3AF' : '#6B7280'}
              transform="rotate(45)"
            >
              {formatTimeLabel(point.timestamp)}
            </text>
          </g>
        );
      });
  };
  
  // Generate y-axis labels and grid lines
  const renderYAxisLabels = () => {
    const tickCount = 5;
    return Array.from({ length: tickCount }).map((_, index) => {
      const value = chartMin + (chartMax - chartMin) * (1 - index / (tickCount - 1));
      const y = scaleY(value);
      
      return (
        <g key={`y-label-${index}`}>
          <line
            x1={padding.left}
            y1={y}
            x2={padding.left + innerWidth}
            y2={y}
            stroke={themeMode === 'dark' ? '#374151' : '#E5E7EB'}
            strokeWidth="1"
            strokeDasharray="3,3"
          />
          <text
            x={padding.left - 10}
            y={y}
            textAnchor="end"
            dominantBaseline="middle"
            fontSize="10"
            fill={themeMode === 'dark' ? '#9CA3AF' : '#6B7280'}
          >
            {value.toFixed(value >= 100 ? 0 : 1)}{unit}
          </text>
        </g>
      );
    });
  };
  
  // Main component render
  return (
    <div 
      className={`
        bg-white dark:bg-gray-800 rounded-lg shadow p-4
        ${isFullScreen ? 'fixed inset-0 z-50 flex flex-col' : ''}
        ${className}
      `}
    >
      {/* Chart header with controls */}
      <div className="flex justify-between items-center mb-4">
        <h3 className="font-bold text-gray-900 dark:text-white">{title}</h3>
        
        {showControls && (
          <div className="flex items-center space-x-2">
            {/* Time range selector */}
            <div className="relative">
              <button
                className="flex items-center space-x-1 px-3 py-1 text-sm border border-gray-300 dark:border-gray-700 rounded hover:bg-gray-50 dark:hover:bg-gray-700"
                onClick={() => setShowTimeRangeDropdown(!showTimeRangeDropdown)}
              >
                <CalendarDays size={14} />
                <span>{timeRanges.find(r => r.value === selectedTimeRange)?.label || selectedTimeRange}</span>
                <ChevronDown size={14} />
              </button>
              
              {showTimeRangeDropdown && (
                <div className="absolute top-full right-0 mt-1 w-32 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded shadow-lg z-10">
                  {timeRanges.map((range) => (
                    <button
                      key={range.value}
                      className={`block w-full text-left px-4 py-2 text-sm ${
                        selectedTimeRange === range.value
                          ? 'bg-indigo-50 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-400'
                          : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
                      }`}
                      onClick={() => {
                        setSelectedTimeRange(range.value);
                        setShowTimeRangeDropdown(false);
                      }}
                    >
                      {range.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
            
            {/* Refresh button */}
            {onRefresh && (
              <button
                className="p-1 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 rounded"
                onClick={handleRefresh}
                disabled={isRefreshing}
              >
                <RefreshCw size={16} className={isRefreshing ? "animate-spin" : ""} />
              </button>
            )}
            
            {/* Download button */}
            <button
              className="p-1 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 rounded"
              onClick={handleDownload}
            >
              <Download size={16} />
            </button>
            
            {/* Fullscreen toggle */}
            <button
              className="p-1 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 rounded"
              onClick={handleToggleFullscreen}
            >
              <Maximize size={16} />
            </button>
          </div>
        )}
      </div>
      
      {/* Chart SVG */}
      <div className={`${isFullScreen ? 'flex-1' : ''} w-full`}>
        <svg
          id={`metrics-chart-${metricName.replace(/\s+/g, '-')}`}
          width="100%"
          height={chartHeight}
          viewBox={`0 0 ${chartWidth} ${chartHeight}`}
          className="overflow-visible"
        >
          {/* Y axis line */}
          <line
            x1={padding.left}
            y1={padding.top}
            x2={padding.left}
            y2={padding.top + innerHeight}
            stroke={themeMode === 'dark' ? '#6B7280' : '#9CA3AF'}
            strokeWidth="1"
          />
          
          {/* X axis line */}
          <line
            x1={padding.left}
            y1={padding.top + innerHeight}
            x2={padding.left + innerWidth}
            y2={padding.top + innerHeight}
            stroke={themeMode === 'dark' ? '#6B7280' : '#9CA3AF'}
            strokeWidth="1"
          />
          
          {/* Y axis labels and grid lines */}
          {renderYAxisLabels()}
          
          {/* X axis labels */}
          {renderXAxisLabels()}
          
          {/* Threshold lines */}
          {renderThresholdLines()}
          
          {/* Area fill under the line */}
          <path
            d={generateAreaPath()}
            fill="url(#gradient)"
            opacity="0.3"
          />
          
          {/* Line */}
          <path
            d={generateLinePath()}
            fill="none"
            stroke="#4F46E5"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="dark:stroke-indigo-400"
          />
          
          {/* Data points */}
          {filteredData.map((point, index) => (
            <circle
              key={`point-${index}`}
              cx={scaleX(index, filteredData.length)}
              cy={scaleY(point.value)}
              r="3"
              fill="#4F46E5"
              stroke={themeMode === 'dark' ? '#111827' : '#FFFFFF'}
              strokeWidth="1"
              className="dark:fill-indigo-400"
            />
          ))}
          
          {/* Gradient for area fill */}
          <defs>
            <linearGradient id="gradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#4F46E5" stopOpacity="0.3" className="dark:stop-color-indigo-400" />
              <stop offset="100%" stopColor="#4F46E5" stopOpacity="0.0" className="dark:stop-color-indigo-400" />
            </linearGradient>
          </defs>
        </svg>
      </div>
      
      {/* Current value and mini stats */}
      {filteredData.length > 0 && (
        <div className="flex items-center justify-between mt-2 text-sm">
          <div>
            <span className="text-gray-500 dark:text-gray-400">Current: </span>
            <span className="font-medium text-gray-900 dark:text-white">
              {filteredData[filteredData.length - 1].value.toFixed(1)}{unit}
            </span>
          </div>
          <div>
            <span className="text-gray-500 dark:text-gray-400">Average: </span>
            <span className="font-medium text-gray-900 dark:text-white">
              {(filteredData.reduce((sum, point) => sum + point.value, 0) / filteredData.length).toFixed(1)}{unit}
            </span>
          </div>
          <div>
            <span className="text-gray-500 dark:text-gray-400">Max: </span>
            <span className="font-medium text-gray-900 dark:text-white">
              {Math.max(...filteredData.map(point => point.value)).toFixed(1)}{unit}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default MetricsChart;
