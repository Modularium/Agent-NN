// monitoring/dashboard/components/common/Alert.tsx
import React from 'react';
import { XCircle, AlertCircle, CheckCircle, Info, X } from 'lucide-react';

type AlertType = 'success' | 'error' | 'warning' | 'info';

interface AlertProps {
  type: AlertType;
  title?: string;
  message: string;
  onClose?: () => void;
  className?: string;
  showIcon?: boolean;
  closable?: boolean;
}

const Alert: React.FC<AlertProps> = ({
  type,
  title,
  message,
  onClose,
  className = '',
  showIcon = true,
  closable = false
}) => {
  // Different styles based on alert type
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
    <div className={`p-4 border rounded-lg ${styles.background} ${styles.border} ${className}`}>
      <div className="flex">
        {showIcon && (
          <div className="flex-shrink-0">
            {styles.icon}
          </div>
        )}
        <div className={`${showIcon ? 'ml-3' : ''} flex-1`}>
          {title && (
            <h3 className={`text-sm font-medium ${styles.titleColor}`}>{title}</h3>
          )}
          <div className={`text-sm ${styles.textColor} ${title ? 'mt-2' : ''}`}>
            {message}
          </div>
        </div>
        {(closable || onClose) && (
          <button
            type="button"
            className={`ml-auto -mx-1.5 -my-1.5 ${styles.textColor} hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg p-1.5`}
            onClick={onClose}
            aria-label="Close"
          >
            <span className="sr-only">Close</span>
            <X className="w-5 h-5" />
          </button>
        )}
      </div>
    </div>
  );
};

export default Alert;
