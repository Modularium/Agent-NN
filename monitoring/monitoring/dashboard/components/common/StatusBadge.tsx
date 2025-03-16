// monitoring/dashboard/components/common/StatusBadge.tsx
import React from 'react';

interface StatusBadgeProps {
  status: string;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
  pulsing?: boolean;
}

const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  className = '',
  size = 'md',
  pulsing = false
}) => {
  // Determine color based on status text
  const getStatusColor = (status: string): string => {
    const statusLower = status.toLowerCase();
    
    if (['active', 'online', 'completed', 'secure', 'success', 'normal'].some(s => statusLower.includes(s))) {
      return 'green';
    }
    if (['warning', 'degraded', 'in-progress', 'running'].some(s => statusLower.includes(s))) {
      return 'yellow';
    }
    if (['error', 'offline', 'failed', 'breach', 'critical'].some(s => statusLower.includes(s))) {
      return 'red';
    }
    if (['idle', 'queued', 'inactive'].some(s => statusLower.includes(s))) {
      return 'gray';
    }
    
    return 'blue'; // Default color
  };

  const color = getStatusColor(status);

  // Color classes mapping
  const colorClasses = {
    green: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300',
    yellow: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300',
    red: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300',
    blue: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300',
    gray: 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300'
  };

  // Size classes mapping
  const sizeClasses = {
    sm: 'px-1.5 py-0.5 text-xs',
    md: 'px-2.5 py-0.5 text-xs',
    lg: 'px-3 py-1 text-sm'
  };

  // Dot color classes
  const dotColorClasses = {
    green: 'bg-green-500 dark:bg-green-400',
    yellow: 'bg-yellow-500 dark:bg-yellow-400',
    red: 'bg-red-500 dark:bg-red-400',
    blue: 'bg-blue-500 dark:bg-blue-400',
    gray: 'bg-gray-500 dark:bg-gray-400'
  };

  const pulsingClass = pulsing ? 'animate-pulse' : '';

  return (
    <span className={`inline-flex items-center rounded-full font-medium ${colorClasses[color]} ${sizeClasses[size]} ${pulsingClass} ${className}`}>
      <span className={`w-2 h-2 rounded-full mr-1.5 ${dotColorClasses[color]}`}></span>
      {status}
    </span>
  );
};

export default StatusBadge;
