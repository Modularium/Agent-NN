// monitoring/dashboard/components/common/StatsCard.tsx
import React from 'react';
import { TrendingUp, TrendingDown, Info } from 'lucide-react';

export interface StatsCardProps {
  title: string;
  value: string | number;
  description?: string;
  trend?: {
    value: number;
    direction: 'up' | 'down';
    isGood: boolean;
    description?: string;
  };
  icon?: React.ReactNode;
  chartComponent?: React.ReactNode;
  loading?: boolean;
  tooltip?: string;
  className?: string;
  color?: 'default' | 'primary' | 'success' | 'warning' | 'danger' | 'info';
  size?: 'sm' | 'md' | 'lg';
  onClick?: () => void;
}

const StatsCard: React.FC<StatsCardProps> = ({
  title,
  value,
  description,
  trend,
  icon,
  chartComponent,
  loading = false,
  tooltip,
  className = '',
  color = 'default',
  size = 'md',
  onClick,
}) => {
  // Card sizing classes
  const sizeClasses = {
    sm: 'p-3',
    md: 'p-4',
    lg: 'p-5',
  };

  // Card color classes
  const colorClasses = {
    default: 'bg-white dark:bg-gray-800',
    primary: 'bg-indigo-50 dark:bg-indigo-900/30',
    success: 'bg-green-50 dark:bg-green-900/30',
    warning: 'bg-yellow-50 dark:bg-yellow-900/30',
    danger: 'bg-red-50 dark:bg-red-900/30',
    info: 'bg-blue-50 dark:bg-blue-900/30',
  };

  // Border color classes
  const borderColorClasses = {
    default: 'border-gray-200 dark:border-gray-700',
    primary: 'border-indigo-200 dark:border-indigo-800',
    success: 'border-green-200 dark:border-green-800',
    warning: 'border-yellow-200 dark:border-yellow-800',
    danger: 'border-red-200 dark:border-red-800',
    info: 'border-blue-200 dark:border-blue-800',
  };

  // Title color classes
  const titleColorClasses = {
    default: 'text-gray-500 dark:text-gray-400',
    primary: 'text-indigo-700 dark:text-indigo-300',
    success: 'text-green-700 dark:text-green-300',
    warning: 'text-yellow-700 dark:text-yellow-300',
    danger: 'text-red-700 dark:text-red-300',
    info: 'text-blue-700 dark:text-blue-300',
  };

  // Value color classes (default is always dark text for better readability)
  const valueColorClasses = {
    default: 'text-gray-900 dark:text-white',
    primary: 'text-indigo-900 dark:text-indigo-100',
    success: 'text-green-900 dark:text-green-100',
    warning: 'text-yellow-900 dark:text-yellow-100',
    danger: 'text-red-900 dark:text-red-100',
    info: 'text-blue-900 dark:text-blue-100',
  };

  // Trend color classes depend on whether trend is good or bad
  const getTrendColorClass = () => {
    if (!trend) return '';

    const isPositiveTrend = (trend.direction === 'up' && trend.isGood) || 
                            (trend.direction === 'down' && !trend.isGood);

    return isPositiveTrend 
      ? 'text-green-600 dark:text-green-400'
      : 'text-red-600 dark:text-red-400';
  };

  // Format trend value
  const formattedTrendValue = trend?.value 
    ? `${trend.value > 0 ? '+' : ''}${trend.value.toLocaleString()}%`
    : '';

  // Add pointer cursor if clickable
  const clickableClass = onClick ? 'cursor-pointer hover:shadow-md transition-shadow' : '';

  return (
    <div
      className={`
        rounded-lg border ${sizeClasses[size]} ${colorClasses[color]} ${borderColorClasses[color]}
        ${clickableClass} ${className}
      `}
      onClick={onClick}
    >
      <div className="flex justify-between items-start">
        <div>
          <div className="flex items-center">
            <h3 className={`text-sm font-medium ${titleColorClasses[color]} mr-1`}>
              {title}
            </h3>
            {tooltip && (
              <div className="relative group">
                <Info size={16} className="text-gray-400 dark:text-gray-500" />
                <div className="absolute z-10 bottom-full mb-2 left-1/2 transform -translate-x-1/2 w-48 p-2 bg-gray-900 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none">
                  {tooltip}
                  <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-2 h-2 bg-gray-900 rotate-45"></div>
                </div>
              </div>
            )}
          </div>
          
          {loading ? (
            <div className="mt-2 animate-pulse">
              <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-24"></div>
            </div>
          ) : (
            <div className={`mt-1 text-2xl font-semibold ${valueColorClasses[color]}`}>
              {value}
            </div>
          )}
          
          {description && (
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              {description}
            </p>
          )}
          
          {trend && (
            <div className={`flex items-center mt-2 ${getTrendColorClass()}`}>
              {trend.direction === 'up' ? (
                <TrendingUp size={16} className="mr-1" />
              ) : (
                <TrendingDown size={16} className="mr-1" />
              )}
              <span className="text-sm font-medium">{formattedTrendValue}</span>
              {trend.description && (
                <span className="ml-1 text-xs">{trend.description}</span>
              )}
            </div>
          )}
        </div>
        
        {icon && (
          <div className={`p-2 rounded-lg bg-opacity-10 dark:bg-opacity-20 ${colorClasses[color]}`}>
            {icon}
          </div>
        )}
      </div>
      
      {chartComponent && (
        <div className="mt-4">
          {chartComponent}
        </div>
      )}
    </div>
  );
};

// Stats card grid component for easily creating grid layouts of stats cards
export interface StatsCardGridProps {
  children: React.ReactNode;
  columns?: 1 | 2 | 3 | 4 | 5 | 6;
  gap?: 'none' | 'xs' | 'sm' | 'md' | 'lg';
  className?: string;
}

export const StatsCardGrid: React.FC<StatsCardGridProps> = ({
  children,
  columns = 4,
  gap = 'md',
  className = '',
}) => {
  // Map columns to grid classes
  const columnsClasses = {
    1: 'grid-cols-1',
    2: 'grid-cols-1 sm:grid-cols-2',
    3: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4',
    5: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5',
    6: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6',
  };

  // Map gap sizes to gap classes
  const gapClasses = {
    none: 'gap-0',
    xs: 'gap-2',
    sm: 'gap-3',
    md: 'gap-4',
    lg: 'gap-6',
  };

  return (
    <div className={`grid ${columnsClasses[columns]} ${gapClasses[gap]} ${className}`}>
      {children}
    </div>
  );
};

export default StatsCard;
