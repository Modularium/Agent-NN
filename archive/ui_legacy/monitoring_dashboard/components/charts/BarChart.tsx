// BarChart.tsx - Reusable bar chart component
import React, { useRef, useEffect } from 'react';
import { useTheme } from '../../context/ThemeContext';

interface BarChartProps {
  title?: string;
  data: {
    labels: string[];
    values: number[];
    colors?: string[];
  };
  height?: number;
  horizontal?: boolean;
  showValues?: boolean;
  showLegend?: boolean;
  showAxis?: boolean;
  maxValue?: number;
  className?: string;
  onClick?: (index: number) => void;
  yAxisLabel?: string;
  xAxisLabel?: string;
}

const BarChart: React.FC<BarChartProps> = ({
  title,
  data,
  height = 300,
  horizontal = false,
  showValues = true,
  showLegend = false,
  showAxis = true,
  maxValue,
  className = '',
  onClick,
  yAxisLabel,
  xAxisLabel,
}) => {
  const { themeMode } = useTheme();
  const svgRef = useRef<SVGSVGElement>(null);
  
  // Default colors if not provided
  const defaultColors = [
    '#4F46E5', // Indigo
    '#10B981', // Green
    '#F59E0B', // Yellow
    '#EF4444', // Red
    '#6366F1', // Indigo/Purple
    '#8B5CF6', // Purple
    '#EC4899', // Pink
    '#14B8A6'  // Teal
  ];
  
  const colors = data.colors || defaultColors;
  
  // Calculate highest value for scaling
  const calculatedMaxValue = maxValue || Math.max(...data.values) * 1.1;
  
  // Padding and chart dimensions
  const padding = { top: 40, right: 20, bottom: 50, left: 60 };
  const chartWidth = horizontal ? 600 : 20 * data.labels.length + padding.left + padding.right;
  const chartHeight = height;
  
  // Chart area dimensions
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;
  
  // Scale factor for bars
  const scale = innerHeight / calculatedMaxValue;
  
  // Bar dimensions
  const barWidth = horizontal 
    ? innerWidth / data.labels.length * 0.7
    : innerHeight / data.labels.length * 0.7;
  const barSpacing = horizontal
    ? innerWidth / data.labels.length
    : innerHeight / data.labels.length;
  
  // Render bar chart
  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow p-4 ${className}`}>
      {title && (
        <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">{title}</h3>
      )}
      
      <svg
        ref={svgRef}
        width="100%"
        height={chartHeight}
        viewBox={`0 0 ${chartWidth} ${chartHeight}`}
        className="overflow-visible"
      >
        {/* Y Axis */}
        {showAxis && (
          <>
            <line
              x1={padding.left}
              y1={padding.top}
              x2={padding.left}
              y2={padding.top + innerHeight}
              stroke={themeMode === 'dark' ? '#6B7280' : '#9CA3AF'}
              strokeWidth="1"
            />
            
            {/* Y Axis ticks */}
            {Array.from({ length: 5 }).map((_, i) => {
              const tickValue = calculatedMaxValue * (1 - i / 4);
              const yPos = padding.top + (innerHeight * i) / 4;
              
              return (
                <g key={`y-tick-${i}`}>
                  <line
                    x1={padding.left - 5}
                    y1={yPos}
                    x2={padding.left}
                    y2={yPos}
                    stroke={themeMode === 'dark' ? '#6B7280' : '#9CA3AF'}
                    strokeWidth="1"
                  />
                  <text
                    x={padding.left - 10}
                    y={yPos}
                    textAnchor="end"
                    dominantBaseline="middle"
                    fontSize="12"
                    fill={themeMode === 'dark' ? '#9CA3AF' : '#6B7280'}
                  >
                    {tickValue.toFixed(tickValue >= 100 ? 0 : 1)}
                  </text>
                </g>
              );
            })}
            
            {/* Y Axis label */}
            {yAxisLabel && (
              <text
                transform={`translate(${padding.left / 3}, ${padding.top + innerHeight / 2}) rotate(-90)`}
                textAnchor="middle"
                fontSize="12"
                fill={themeMode === 'dark' ? '#9CA3AF' : '#6B7280'}
              >
                {yAxisLabel}
              </text>
            )}
            
            {/* X Axis */}
            <line
              x1={padding.left}
              y1={padding.top + innerHeight}
              x2={padding.left + innerWidth}
              y2={padding.top + innerHeight}
              stroke={themeMode === 'dark' ? '#6B7280' : '#9CA3AF'}
              strokeWidth="1"
            />
            
            {/* X Axis label */}
            {xAxisLabel && (
              <text
                x={padding.left + innerWidth / 2}
                y={padding.top + innerHeight + 40}
                textAnchor="middle"
                fontSize="12"
                fill={themeMode === 'dark' ? '#9CA3AF' : '#6B7280'}
              >
                {xAxisLabel}
              </text>
            )}
          </>
        )}
        
        {/* Bars */}
        {data.values.map((value, index) => {
          // If horizontal, bars grow from left to right
          // If vertical, bars grow from bottom to top
          const barHeight = value * scale;
          const x = horizontal
            ? padding.left
            : padding.left + index * barSpacing;
          const y = horizontal
            ? padding.top + index * barSpacing
            : padding.top + innerHeight - barHeight;
          const width = horizontal ? barHeight : barWidth;
          const height = horizontal ? barWidth : barHeight;
          
          return (
            <g key={`bar-${index}`} onClick={() => onClick && onClick(index)}>
              <rect
                x={x}
                y={y}
                width={width}
                height={height}
                fill={colors[index % colors.length]}
                rx={4}
                className="transition-all duration-300 hover:opacity-80 cursor-pointer"
              />
              
              {/* Bar labels */}
              {horizontal ? (
                <text
                  x={x + barHeight + 5}
                  y={y + barWidth / 2}
                  dominantBaseline="middle"
                  fontSize="12"
                  fill={themeMode === 'dark' ? '#E5E7EB' : '#374151'}
                >
                  {showValues && value.toFixed(1)}
                </text>
              ) : (
                <text
                  x={x + barWidth / 2}
                  y={y - 5}
                  textAnchor="middle"
                  fontSize="12"
                  fill={themeMode === 'dark' ? '#E5E7EB' : '#374151'}
                >
                  {showValues && value.toFixed(1)}
                </text>
              )}
              
              {/* X-axis labels for vertical chart, Y-axis labels for horizontal chart */}
              {horizontal ? (
                <text
                  x={padding.left - 10}
                  y={y + barWidth / 2}
                  textAnchor="end"
                  dominantBaseline="middle"
                  fontSize="12"
                  fill={themeMode === 'dark' ? '#9CA3AF' : '#6B7280'}
                >
                  {data.labels[index]}
                </text>
              ) : (
                <text
                  x={x + barWidth / 2}
                  y={padding.top + innerHeight + 15}
                  textAnchor="middle"
                  fontSize="12"
                  fill={themeMode === 'dark' ? '#9CA3AF' : '#6B7280'}
                  transform={`rotate(45, ${x + barWidth / 2}, ${padding.top + innerHeight + 15})`}
                >
                  {data.labels[index]}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      
      {/* Legend */}
      {showLegend && (
        <div className="flex flex-wrap gap-4 mt-4 justify-center">
          {data.labels.map((label, index) => (
            <div key={`legend-${index}`} className="flex items-center">
              <div
                className="w-3 h-3 mr-1"
                style={{ backgroundColor: colors[index % colors.length] }}
              ></div>
              <span className="text-xs text-gray-600 dark:text-gray-400">{label}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default BarChart;
