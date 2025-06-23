// monitoring/dashboard/components/charts/InteractiveLineChart.tsx
import React, { useState, useEffect, useRef } from 'react';
import { useTheme } from '../../context/ThemeContext';
import { ChevronDown, Maximize, Download, RefreshCw } from 'lucide-react';
import { TimeSeriesData } from '../../types/metrics';
import { formatRelativeTime } from '../../utils/formatters';

interface InteractiveLineChartProps {
  title: string;
  data: TimeSeriesData[];
  yAxisLabel?: string;
  xAxisLabel?: string;
  color?: string;
  height?: number;
  showControls?: boolean;
  showLegend?: boolean;
  showTooltip?: boolean;
  allowZoom?: boolean;
  allowDownload?: boolean;
  refreshInterval?: number; // in seconds
  onRefresh?: () => void;
  compareData?: {
    label: string;
    data: TimeSeriesData[];
    color: string;
  }[];
  timeRange?: '1h' | '6h' | '24h' | '7d' | '30d' | 'all';
  onTimeRangeChange?: (range: '1h' | '6h' | '24h' | '7d' | '30d' | 'all') => void;
}

const InteractiveLineChart: React.FC<InteractiveLineChartProps> = ({
  title,
  data,
  yAxisLabel = '',
  xAxisLabel = 'Time',
  color = '#4F46E5', // Indigo color
  height = 300,
  showControls = true,
  showLegend = true,
  showTooltip = true,
  allowZoom = true,
  allowDownload = true,
  refreshInterval,
  onRefresh,
  compareData = [],
  timeRange = 'all',
  onTimeRangeChange
}) => {
  const { themeMode } = useTheme();
  const [isFullScreen, setIsFullScreen] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1h' | '6h' | '24h' | '7d' | '30d' | 'all'>(timeRange);
  const [tooltipVisible, setTooltipVisible] = useState(false);
  const [tooltipData, setTooltipData] = useState<{ x: number; y: number; value: number; timestamp: string } | null>(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [filteredData, setFilteredData] = useState<TimeSeriesData[]>(data);
  const [filteredCompareData, setFilteredCompareData] = useState<typeof compareData>(compareData);
  const svgRef = useRef<SVGSVGElement>(null);

  // Define chart dimensions
  const margin = { top: 20, right: 30, bottom: 40, left: 50 };
  const chartWidth = isFullScreen ? window.innerWidth - 100 : 800;
  const chartHeight = isFullScreen ? window.innerHeight - 200 : height;
  const width = chartWidth - margin.left - margin.right;
  const height = chartHeight - margin.top - margin.bottom;

  // Auto-refresh timer
  useEffect(() => {
    if (refreshInterval && onRefresh) {
      const timer = setInterval(() => {
        onRefresh();
      }, refreshInterval * 1000);
      
      return () => clearInterval(timer);
    }
  }, [refreshInterval, onRefresh]);

  // Filter data based on time range
  useEffect(() => {
    if (timeRange !== selectedTimeRange) {
      setSelectedTimeRange(timeRange);
    }
    
    const filterByTimeRange = (data: TimeSeriesData[]) => {
      if (selectedTimeRange === 'all') {
        return data;
      }
      
      const now = new Date();
      let cutoffTime = new Date();
      
      switch (selectedTimeRange) {
        case '1h':
          cutoffTime.setHours(now.getHours() - 1);
          break;
        case '6h':
          cutoffTime.setHours(now.getHours() - 6);
          break;
        case '24h':
          cutoffTime.setHours(now.getHours() - 24);
          break;
        case '7d':
          cutoffTime.setDate(now.getDate() - 7);
          break;
        case '30d':
          cutoffTime.setDate(now.getDate() - 30);
          break;
      }
      
      return data.filter(item => new Date(item.timestamp) >= cutoffTime);
    };
    
    setFilteredData(filterByTimeRange(data));
    setFilteredCompareData(compareData.map(item => ({
      ...item,
      data: filterByTimeRange(item.data)
    })));
  }, [data, compareData, selectedTimeRange, timeRange]);

  // Handle time range change
  const handleTimeRangeChange = (range: '1h' | '6h' | '24h' | '7d' | '30d' | 'all') => {
    setSelectedTimeRange(range);
    if (onTimeRangeChange) {
      onTimeRangeChange(range);
    }
  };

  // Function to handle refresh
  const handleRefresh = async () => {
    if (onRefresh) {
      setIsRefreshing(true);
      await onRefresh();
      setIsRefreshing(false);
    }
  };

  // Function to handle full screen toggle
  const toggleFullScreen = () => {
    setIsFullScreen(!isFullScreen);
  };

  // Function to download chart as SVG
  const downloadSVG = () => {
    if (!svgRef.current) return;
    
    const svgData = new XMLSerializer().serializeToString(svgRef.current);
    const blob = new Blob([svgData], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `${title.replace(/\s+/g, '-').toLowerCase()}-chart.svg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Calculate scales
  const getXScale = (data: TimeSeriesData[]) => {
    if (!data.length) return { min: 0, max: 100, scale: (x: number) => x };
    
    const timestamps = data.map(d => new Date(d.timestamp).getTime());
    const min = Math.min(...timestamps);
    const max = Math.max(...timestamps);
    const range = max - min;
    
    return {
      min,
      max,
      scale: (x: number) => ((x - min) / range) * width * zoomLevel + panOffset.x
    };
  };

  const getYScale = (data: TimeSeriesData[]) => {
    if (!data.length) return { min: 0, max: 100, scale: (y: number) => height - y };
    
    const values = data.map(d => d.value);
    let min = Math.min(...values);
    let max = Math.max(...values);
    
    // Add some padding
    const padding = (max - min) * 0.1;
    min = Math.max(0, min - padding);
    max = max + padding;
    
    // Ensure min and max are different
    if (min === max) {
      min = min > 0 ? min * 0.9 : 0;
      max = max === 0 ? 1 : max * 1.1;
    }
    
    return {
      min,
      max,
      scale: (y: number) => height - ((y - min) / (max - min)) * height * zoomLevel + panOffset.y
    };
  };

  // Mouse event handlers for zooming and panning
  const handleMouseDown = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!allowZoom) return;
    
    setIsDragging(true);
    setDragStart({
      x: e.clientX,
      y: e.clientY
    });
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!allowZoom || !isDragging) return;
    
    const dx = e.clientX - dragStart.x;
    const dy = e.clientY - dragStart.y;
    
    setPanOffset({
      x: panOffset.x + dx,
      y: panOffset.y + dy
    });
    
    setDragStart({
      x: e.clientX,
      y: e.clientY
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleMouseLeave = () => {
    setIsDragging(false);
    setTooltipVisible(false);
  };

  const handleWheel = (e: React.WheelEvent<SVGSVGElement>) => {
    if (!allowZoom) return;
    
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.1 : -0.1;
    setZoomLevel(Math.max(0.5, Math.min(5, zoomLevel + delta)));
  };

  // Tooltip handler
  const handleMouseOverPoint = (datum: TimeSeriesData, x: number, y: number) => {
    if (!showTooltip) return;
    
    setTooltipData({
      x,
      y,
      value: datum.value,
      timestamp: datum.timestamp
    });
    setTooltipVisible(true);
  };

  const resetZoom = () => {
    setZoomLevel(1);
    setPanOffset({ x: 0, y: 0 });
  };

  // Prepare data for rendering
  const xScale = getXScale([...filteredData, ...filteredCompareData.flatMap(c => c.data)]);
  const yScale = getYScale([...filteredData, ...filteredCompareData.flatMap(c => c.data)]);

  // Generate path for the line
  const generatePath = (data: TimeSeriesData[], color: string) => {
    if (!data.length) return null;
    
    const pathData = data
      .map((d, i) => {
        const x = xScale.scale(new Date(d.timestamp).getTime());
        const y = yScale.scale(d.value);
        return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
      })
      .join(' ');
    
    return (
      <>
        <path
          d={pathData}
          fill="none"
          stroke={color}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {data.map((d, i) => (
          <circle
            key={i}
            cx={xScale.scale(new Date(d.timestamp).getTime())}
            cy={yScale.scale(d.value)}
            r="4"
            fill={color}
            stroke={themeMode === 'dark' ? '#111827' : '#FFFFFF'}
            strokeWidth="1"
            className="cursor-pointer"
            onMouseOver={() => handleMouseOverPoint(
              d, 
              xScale.scale(new Date(d.timestamp).getTime()),
              yScale.scale(d.value)
            )}
            onMouseLeave={() => setTooltipVisible(false)}
          />
        ))}
      </>
    );
  };

  // Generate x-axis ticks
  const xAxisTicks = () => {
    if (!filteredData.length) return null;
    
    const timestamps = [...filteredData, ...filteredCompareData.flatMap(c => c.data)]
      .map(d => new Date(d.timestamp).getTime())
      .sort((a, b) => a - b);
    
    // Determine number of ticks based on chart width
    const numTicks = Math.min(Math.floor(width / 100), 10);
    
    // Generate evenly spaced ticks
    const ticks = [];
    for (let i = 0; i < numTicks; i++) {
      const percent = i / (numTicks - 1);
      const index = Math.floor(percent * (timestamps.length - 1));
      if (index < timestamps.length) {
        ticks.push(timestamps[index]);
      }
    }
    
    return ticks.map((timestamp, i) => {
      const x = xScale.scale(timestamp);
      const date = new Date(timestamp);
      
      // Format the label based on time range
      let label = '';
      switch (selectedTimeRange) {
        case '1h':
        case '6h':
          label = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
          break;
        case '24h':
          label = date.toLocaleTimeString([], { hour: '2-digit' });
          break;
        case '7d':
          label = date.toLocaleDateString([], { month: 'short', day: 'numeric' });
          break;
        case '30d':
        case 'all':
          label = date.toLocaleDateString([], { month: 'short', day: 'numeric' });
          break;
      }
      
      return (
        <g key={i} transform={`translate(${x}, ${height + 10})`}>
          <line y2="6" stroke={themeMode === 'dark' ? '#4B5563' : '#9CA3AF'} />
          <text
            dy="0.71em"
            y="9"
            textAnchor="middle"
            fill={themeMode === 'dark' ? '#9CA3AF' : '#4B5563'}
            fontSize="10"
          >
            {label}
          </text>
        </g>
      );
    });
  };

  // Generate y-axis ticks
  const yAxisTicks = () => {
    const numTicks = 5;
    const ticks = [];
    
    for (let i = 0; i < numTicks; i++) {
      const value = yScale.min + ((yScale.max - yScale.min) * i) / (numTicks - 1);
      ticks.push(value);
    }
    
    return ticks.map((value, i) => {
      const y = yScale.scale(value);
      
      return (
        <g key={i} transform={`translate(0, ${y})`}>
          <line x2="-6" stroke={themeMode === 'dark' ? '#4B5563' : '#9CA3AF'} />
          <text
            x="-10"
            dy="0.32em"
            textAnchor="end"
            fill={themeMode === 'dark' ? '#9CA3AF' : '#4B5563'}
            fontSize="10"
          >
            {value.toFixed(1)}
          </text>
          <line
            x1="0"
            x2={width}
            stroke={themeMode === 'dark' ? '#374151' : '#E5E7EB'}
            strokeDasharray="3,3"
          />
        </g>
      );
    });
  };

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow ${isFullScreen ? 'fixed inset-0 z-50 p-4' : ''}`}>
      {/* Chart Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <h3 className="font-bold text-gray-900 dark:text-white">{title}</h3>
          
          {showControls && (
            <div className="flex items-center space-x-2">
              {/* Time range selector */}
              <div className="relative">
                <button
                  className="flex items-center space-x-1 px-2 py-1 text-sm border border-gray-300 dark:border-gray-700 rounded hover:bg-gray-50 dark:hover:bg-gray-700"
                >
                  <span>{selectedTimeRange === 'all' ? 'All time' : selectedTimeRange}</span>
                  <ChevronDown size={14} />
                </button>
                <div className="absolute right-0 mt-1 w-32 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded shadow-lg z-10 hidden group-hover:block">
                  <div className="py-1">
                    {(['1h', '6h', '24h', '7d', '30d', 'all'] as const).map((range) => (
                      <button
                        key={range}
                        className={`block w-full text-left px-4 py-2 text-sm ${
                          selectedTimeRange === range
                            ? 'bg-indigo-50 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-400'
                            : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
                        }`}
                        onClick={() => handleTimeRangeChange(range)}
                      >
                        {range === 'all' ? 'All time' : range}
                      </button>
                    ))}
                  </div>
                </div>
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
              
              {/* Full screen button */}
              <button
                className="p-1 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 rounded"
                onClick={toggleFullScreen}
              >
                <Maximize size={16} />
              </button>
              
              {/* Download button */}
              {allowDownload && (
                <button
                  className="p-1 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 rounded"
                  onClick={downloadSVG}
                >
                  <Download size={16} />
                </button>
              )}
            </div>
          )}
        </div>
        
        {/* Last updated info */}
        {filteredData.length > 0 && (
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Last updated: {formatRelativeTime(filteredData[filteredData.length - 1].timestamp)}
          </div>
        )}
        
        {/* Legend */}
        {showLegend && (
          <div className="flex items-center mt-2 space-x-4">
            <div className="flex items-center">
              <div
                className="w-3 h-3 rounded-full mr-1"
                style={{ backgroundColor: color }}
              ></div>
              <span className="text-xs text-gray-600 dark:text-gray-400">
                {yAxisLabel || 'Value'}
              </span>
            </div>
            
            {filteredCompareData.map((item, i) => (
              <div key={i} className="flex items-center">
                <div
                  className="w-3 h-3 rounded-full mr-1"
                  style={{ backgroundColor: item.color }}
                ></div>
                <span className="text-xs text-gray-600 dark:text-gray-400">
                  {item.label}
                </span>
              </div>
            ))}
            
            {allowZoom && (
              <button
                className="text-xs text-indigo-600 dark:text-indigo-400 ml-auto"
                onClick={resetZoom}
              >
                Reset zoom
              </button>
            )}
          </div>
        )}
      </div>
      
      {/* Chart SVG */}
      <div className="p-4">
        <svg
          ref={svgRef}
          width={chartWidth}
          height={chartHeight}
          className="chart-svg"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
          onWheel={handleWheel}
        >
          <g transform={`translate(${margin.left}, ${margin.top})`}>
            {/* X and Y axes */}
            <line
              x1="0"
              y1={height}
              x2={width}
              y2={height}
              stroke={themeMode === 'dark' ? '#4B5563' : '#9CA3AF'}
            />
            <line
              x1="0"
              y1="0"
              x2="0"
              y2={height}
              stroke={themeMode === 'dark' ? '#4B5563' : '#9CA3AF'}
            />
            
            {/* X and Y axis labels */}
            <text
              textAnchor="middle"
              x={width / 2}
              y={height + 35}
              fill={themeMode === 'dark' ? '#9CA3AF' : '#4B5563'}
              fontSize="12"
            >
              {xAxisLabel}
            </text>
            <text
              textAnchor="middle"
              transform={`translate(-35, ${height / 2}) rotate(-90)`}
              fill={themeMode === 'dark' ? '#9CA3AF' : '#4B5563'}
              fontSize="12"
            >
              {yAxisLabel}
            </text>
            
            {/* Grid lines */}
            <g className="grid">{yAxisTicks()}</g>
            <g className="x-axis">{xAxisTicks()}</g>
            
            {/* Data visualization */}
            {generatePath(filteredData, color)}
            
            {/* Comparison data */}
            {filteredCompareData.map((item, i) => (
              <React.Fragment key={i}>
                {generatePath(item.data, item.color)}
              </React.Fragment>
            ))}
            
            {/* Tooltip */}
            {showTooltip && tooltipVisible && tooltipData && (
              <g transform={`translate(${tooltipData.x}, ${tooltipData.y})`}>
                <rect
                  x="10"
                  y="-35"
                  width="140"
                  height="30"
                  rx="3"
                  fill={themeMode === 'dark' ? '#1F2937' : '#FFFFFF'}
                  stroke={themeMode === 'dark' ? '#374151' : '#E5E7EB'}
                />
                <text
                  x="15"
                  y="-15"
                  fill={themeMode === 'dark' ? '#F9FAFB' : '#111827'}
                  fontSize="12"
                >
                  <tspan x="15" dy="-5">
                    {yAxisLabel}: {tooltipData.value.toFixed(2)}
                  </tspan>
                  <tspan x="15" dy="15">
                    {formatRelativeTime(tooltipData.timestamp)}
                  </tspan>
                </text>
              </g>
            )}
          </g>
        </svg>
      </div>
    </div>
  );
};

export default InteractiveLineChart;
