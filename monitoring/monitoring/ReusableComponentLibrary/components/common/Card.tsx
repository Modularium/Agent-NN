// monitoring/dashboard/components/common/Card.tsx
import React, { ReactNode } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';

interface CardProps {
  title?: string;
  children: ReactNode;
  className?: string;
  headerAction?: ReactNode;
  footer?: ReactNode;
  noPadding?: boolean;
  hoverable?: boolean;
  bordered?: boolean;
  collapsible?: boolean;
  defaultCollapsed?: boolean;
}

const Card: React.FC<CardProps> = ({
  title,
  children,
  className = '',
  headerAction,
  footer,
  noPadding = false,
  hoverable = false,
  bordered = true,
  collapsible = false,
  defaultCollapsed = false
}) => {
  const [isCollapsed, setIsCollapsed] = React.useState(defaultCollapsed);
  return (
    <div 
      className={`
        bg-white dark:bg-gray-800 
        shadow rounded-lg 
        ${bordered ? 'border border-gray-200 dark:border-gray-700' : ''} 
        ${hoverable ? 'transition-shadow duration-200 hover:shadow-md' : ''} 
        ${className}
      `}
    >
      {title && (
        <div className="flex justify-between items-center border-b border-gray-200 dark:border-gray-700 p-4">
          <h3 className="font-bold text-gray-800 dark:text-white">{title}</h3>
          <div className="flex items-center space-x-2">
            {headerAction}
            {collapsible && (
              <button 
                onClick={() => setIsCollapsed(!isCollapsed)}
                className="p-1 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                {isCollapsed ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
              </button>
            )}
          </div>
        </div>
      )}
      
      <div className={`${noPadding ? '' : 'p-4'} ${isCollapsed ? 'hidden' : ''}`}>
        {children}
      </div>
      
      {footer && (
        <div className="border-t border-gray-200 dark:border-gray-700 p-4">
          {footer}
        </div>
      )}
    </div>
  );
};

export default Card;

// monitoring/dashboard/components/common/LoadingSpinner.tsx
import React from 'react';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  text?: string;
  fullPage?: boolean;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  text = 'Loading...',
  fullPage = false
}) => {
  const sizeClasses = {
    sm: 'h-4 w-4',
    md: 'h-8 w-8',
    lg: 'h-12 w-12'
  };

  const spinner = (
    <div className="flex flex-col items-center justify-center">
      <div className={`${sizeClasses[size]} border-t-2 border-b-2 border-indigo-600 rounded-full animate-spin`}></div>
      {text && <p className="mt-2 text-gray-600 dark:text-gray-300">{text}</p>}
    </div>
  );

  if (fullPage) {
    return (
      <div className="fixed inset-0 flex items-center justify-center bg-white bg-opacity-75 dark:bg-gray-900 dark:bg-opacity-75 z-50">
        {spinner}
      </div>
    );
  }

  return spinner;
};

export default LoadingSpinner;

// monitoring/dashboard/components/common/ErrorBoundary.tsx
import React, { Component, ErrorInfo, ReactNode } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { 
      hasError: false,
      error: null
    };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
    
    // Log error to monitoring service
    console.error('Error caught by ErrorBoundary:', error, errorInfo);
  }

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }
      
      return (
        <div className="p-4 border border-red-300 bg-red-50 dark:bg-red-900/20 rounded-lg">
          <h3 className="text-lg font-medium text-red-800 dark:text-red-300">Something went wrong</h3>
          <p className="mt-2 text-sm text-red-700 dark:text-red-400">
            {this.state.error?.message || 'An unexpected error occurred'}
          </p>
          <button
            className="mt-3 text-sm font-medium text-indigo-600 dark:text-indigo-400 hover:text-indigo-500"
            onClick={() => this.setState({ hasError: false, error: null })}
          >
            Try again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;

// monitoring/dashboard/components/common/Alert.tsx
import React from 'react';
import { XCircle, AlertCircle, CheckCircle, Info } from 'lucide-react';

type AlertType = 'success' | 'error' | 'warning' | 'info';

interface AlertProps {
  type: AlertType;
  title?: string;
  message: string;
  onClose?: () => void;
}

const Alert: React.FC<AlertProps> = ({ type, title, message, onClose }) => {
  const typeStyles = {
    success: {
      background: 'bg-green-50 dark:bg-green-900/20',
      border: 'border-green-300 dark:border-green-800',
      titleColor: 'text-green-800 dark:text-green-300',
      textColor: 'text-green-700 dark:text-green-400',
      icon: <CheckCircle className="w-5 h-5 text-green-500" />
    },
    error: {
      background: 'bg-red-50 dark:bg-red-900/20',
      border: 'border-red-300 dark:border-red-800',
      titleColor: 'text-red-800 dark:text-red-300',
      textColor: 'text-red-700 dark:text-red-400',
      icon: <XCircle className="w-5 h-5 text-red-500" />
    },
    warning: {
      background: 'bg-yellow-50 dark:bg-yellow-900/20',
      border: 'border-yellow-300 dark:border-yellow-800',
      titleColor: 'text-yellow-800 dark:text-yellow-300',
      textColor: 'text-yellow-700 dark:text-yellow-400',
      icon: <AlertCircle className="w-5 h-5 text-yellow-500" />
    },
    info: {
      background: 'bg-blue-50 dark:bg-blue-900/20',
      border: 'border-blue-300 dark:border-blue-800',
      titleColor: 'text-blue-800 dark:text-blue-300',
      textColor: 'text-blue-700 dark:text-blue-400',
      icon: <Info className="w-5 h-5 text-blue-500" />
    }
  };

  const styles = typeStyles[type];

  return (
    <div className={`p-4 border rounded-lg ${styles.background} ${styles.border}`}>
      <div className="flex">
        <div className="flex-shrink-0">
          {styles.icon}
        </div>
        <div className="ml-3 flex-1">
          {title && (
            <h3 className={`text-sm font-medium ${styles.titleColor}`}>{title}</h3>
          )}
          <div className={`text-sm ${styles.textColor} ${title ? 'mt-2' : ''}`}>
            {message}
          </div>
        </div>
        {onClose && (
          <button
            type="button"
            className={`ml-auto -mx-1.5 -my-1.5 ${styles.textColor} hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg p-1.5`}
            onClick={onClose}
            aria-label="Close"
          >
            <span className="sr-only">Close</span>
            <XCircle className="w-5 h-5" />
          </button>
        )}
      </div>
    </div>
  );
};

export default Alert;

// monitoring/dashboard/components/common/MetricCard.tsx
import React from 'react';
import { ArrowUp, ArrowDown } from 'lucide-react';
import Card from './Card';
import { getStatusColor } from '../../utils/formatters';

interface MetricCardProps {
  title: string;
  value: number | string;
  unit?: string;
  trend?: {
    value: number;
    isUpward: boolean;
    isGood: boolean;
  };
  status?: 'normal' | 'warning' | 'critical';
  icon?: React.ReactNode;
  className?: string;
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  unit = '',
  trend,
  status = 'normal',
  icon,
  className = ''
}) => {
  // Determine status color
  const statusColors = {
    normal: 'bg-green-500',
    warning: 'bg-yellow-500',
    critical: 'bg-red-500'
  };

  // Determine trend color
  const getTrendColor = (isUpward: boolean, isGood: boolean) => {
    if ((isUpward && isGood) || (!isUpward && !isGood)) {
      return 'text-green-500';
    }
    return 'text-red-500';
  };

  return (
    <Card className={`h-full ${className}`}>
      <div className="flex items-center justify-between">
        <div className="flex flex-col">
          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium text-gray-500 dark:text-gray-400">{title}</span>
            {status && (
              <span className={`inline-block w-2 h-2 rounded-full ${statusColors[status]}`}></span>
            )}
          </div>
          <div className="flex items-baseline mt-1">
            <span className="text-2xl font-semibold text-gray-900 dark:text-white">
              {value}
            </span>
            {unit && (
              <span className="ml-1 text-sm text-gray-500 dark:text-gray-400">{unit}</span>
            )}
          </div>
          {trend && (
            <div className={`flex items-center mt-1 ${getTrendColor(trend.isUpward, trend.isGood)}`}>
              {trend.isUpward ? (
                <ArrowUp className="w-4 h-4 mr-1" />
              ) : (
                <ArrowDown className="w-4 h-4 mr-1" />
              )}
              <span className="text-sm">{trend.value}%</span>
            </div>
          )}
        </div>
        {icon && (
          <div className="p-2 bg-indigo-100 dark:bg-indigo-900/20 rounded-lg">
            {icon}
          </div>
        )}
      </div>
    </Card>
  );
};

export default MetricCard;

// monitoring/dashboard/components/common/StatusBadge.tsx
import React from 'react';
import { getStatusColor } from '../../utils/formatters';

interface StatusBadgeProps {
  status: string;
  className?: string;
}

const StatusBadge: React.FC<StatusBadgeProps> = ({ status, className = '' }) => {
  const statusColor = getStatusColor(status);
  
  const colorClasses = {
    green: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300',
    yellow: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300',
    red: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300',
    gray: 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300',
    blue: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300'
  };

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${colorClasses[statusColor]} ${className}`}>
      <span className={`w-2 h-2 rounded-full bg-${statusColor}-500 mr-1.5 dark:bg-${statusColor}-400`}></span>
      {status}
    </span>
  );
};

export default StatusBadge;
