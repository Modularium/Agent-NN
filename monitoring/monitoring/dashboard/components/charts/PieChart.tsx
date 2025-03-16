// PieChart.tsx - Reusable pie chart component
import React, { useState } from 'react';
import { useTheme } from '../../context/ThemeContext';

interface PieChartProps {
  title?: string;
  data: {
    labels: string[];
    values: number[];
    colors?: string[];
  };
  size?: number;
  showLabels?: boolean;
  showLegend?: boolean;
  showPercentages?: boolean;
  donut?: boolean;
  donutThickness?: number;
  className?: string;
  onClick?: (index: number) => void;
  centerLabel?: string;
}

const PieChart: React.FC<PieChartProps> = ({
  title,
  data,
  size = 200,
  showLabels = false,
  showLegend = true,
  showPercentages = true,
  donut = false,
  donutThickness = 50,
  className = '',
  onClick,
  centerLabel,
}) => {
  const { themeMode } = useTheme();
  const [hoveredSlice, setHoveredSlice] = useState<number | null>(null);
  
  // Default colors if not provided
  const defaultColors = [
    '#4F46E5', // Indigo
    '#10B981', // Green
    '#F59E0B', // Yellow
    '#EF4444', // Red
    '#6366F1', // Indigo/Purple
    '#8B5CF6', // Purple
    '#EC4899', // Pink
    '#14B8A6', // Teal
  ];
  
  const colors = data.colors || defaultColors;
  
  // Calculate the sum of all values
  const total = data.values.reduce((sum, value) => sum + value, 0);
  
  // Helper function to calculate the coordinates on the circle
  const getCoordinatesForPercent = (percent: number) => {
    const x = Math.cos(2 * Math.PI * percent);
    const y = Math.sin(2 * Math.PI * percent);
    return [x, y];
  };
  
  // Create pie wedges
  const slices = data.values.map((value, index) => {
    const percentage = value / total;
    
    // We compute the start and end angles for this slice
    const startPercent = data.values.slice(0, index).reduce((sum, v) => sum + v / total, 0);
    const endPercent = startPercent + percentage;
    
    // Calculate coordinates
    const [startX, startY] = getCoordinatesForPercent(startPercent);
    const [endX, endY] = getCoordinatesForPercent(endPercent);
    
    // Determine if the slice is large (more than 180 degrees or π radians)
    const largeArcFlag = endPercent - startPercent > 0.5 ? 1 : 0;
    
    // Radius for the pie chart
    const radius = size / 2;
    const innerRadius = donut ? radius - donutThickness : 0;
    
    // Slight adjustment for hovered slices
    const hoverOffset = hoveredSlice === index ? 10 : 0;
    const adjustedRadius = radius + hoverOffset;
    
    // Calculate the SVG path for the slice
    // For a donut chart, we need to draw the outer and inner arcs
    let pathData;
    
    if (donut) {
      // Calculate inner arc coordinates
      const [innerStartX, innerStartY] = [startX * innerRadius, startY * innerRadius];
      const [innerEndX, innerEndY] = [endX * innerRadius, endY * innerRadius];
      
      // Outer arc
      pathData = [
        `M ${startX * adjustedRadius + radius} ${startY * adjustedRadius + radius}`, // Move to start point on outer edge
        `A ${adjustedRadius} ${adjustedRadius} 0 ${largeArcFlag} 1 ${endX * adjustedRadius + radius} ${endY * adjustedRadius + radius}`, // Outer arc
        `L ${endX * innerRadius + radius} ${endY * innerRadius + radius}`, // Line to inner edge
        `A ${innerRadius} ${innerRadius} 0 ${largeArcFlag} 0 ${startX * innerRadius + radius} ${startY * innerRadius + radius}`, // Inner arc
        `Z` // Close path
      ].join(' ');
    } else {
      // Simple pie slice
      pathData = [
        `M ${radius} ${radius}`, // Move to center
        `L ${startX * adjustedRadius + radius} ${startY * adjustedRadius + radius}`, // Line to start point on edge
        `A ${adjustedRadius} ${adjustedRadius} 0 ${largeArcFlag} 1 ${endX * adjustedRadius + radius} ${endY * adjustedRadius + radius}`, // Arc
        `Z` // Close path back to center
      ].join(' ');
    }
    
    return {
      path: pathData,
      percentage,
      color: colors[index % colors.length],
      index,
      value,
      label: data.labels[index],
      // Calculate position for the label
      labelPosition: {
        // Position halfway through the slice, a bit away from center
        x: (startX + endX) / 2 * radius * 0.65 + radius,
        y: (startY + endY) / 2 * radius * 0.65 + radius
      }
    };
  });
  
  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow p-4 ${className}`}>
      {title && (
        <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">{title}</h3>
      )}
      
      <div className="flex flex-col items-center">
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          className="overflow-visible"
        >
          {/* Render each slice */}
          {slices.map((slice) => (
            <g 
              key={`slice-${slice.index}`}
              onMouseEnter={() => setHoveredSlice(slice.index)}
              onMouseLeave={() => setHoveredSlice(null)}
              onClick={() => onClick && onClick(slice.index)}
              className="transition-all duration-300 cursor-pointer"
            >
              <path
                d={slice.path}
                fill={slice.color}
                stroke={themeMode === 'dark' ? '#111827' : '#FFFFFF'}
                strokeWidth="1"
                className="transition-all duration-300 hover:opacity-80"
              />
              
              {/* Labels inside slices */}
              {showLabels && slice.percentage > 0.05 && (
                <text
                  x={slice.labelPosition.x}
                  y={slice.labelPosition.y}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fontSize="12"
                  fill="#FFFFFF"
                  pointerEvents="none"
                >
                  {showPercentages 
                    ? `${(slice.percentage * 100).toFixed(0)}%` 
                    : slice.label}
                </text>
              )}
            </g>
          ))}
          
          {/* Center label for donut chart */}
          {donut && centerLabel && (
            <text
              x={size / 2}
              y={size / 2}
              textAnchor="middle"
              dominantBaseline="middle"
              fontSize="14"
              fontWeight="bold"
              fill={themeMode === 'dark' ? '#E5E7EB' : '#374151'}
            >
              {centerLabel}
            </text>
          )}
        </svg>
        
        {/* Legend */}
        {showLegend && (
          <div className="flex flex-wrap gap-x-4 gap-y-2 mt-4 justify-center">
            {data.labels.map((label, index) => (
              <div 
                key={`legend-${index}`} 
                className="flex items-center"
                onMouseEnter={() => setHoveredSlice(index)}
                onMouseLeave={() => setHoveredSlice(null)}
              >
                <div
                  className="w-3 h-3 mr-1"
                  style={{ backgroundColor: colors[index % colors.length] }}
                ></div>
                <span className="text-xs text-gray-600 dark:text-gray-400">
                  {label}
                  {showPercentages && (
                    <span className="ml-1">
                      ({((data.values[index] / total) * 100).toFixed(1)}%)
                    </span>
                  )}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default PieChart;
