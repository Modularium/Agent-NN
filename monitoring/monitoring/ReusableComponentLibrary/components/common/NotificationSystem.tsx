// monitoring/dashboard/components/common/NotificationSystem.tsx
import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { XCircle, CheckCircle, AlertCircle, Info, Bell, X } from 'lucide-react';

// Types
export type NotificationType = 'success' | 'error' | 'warning' | 'info';

export interface Notification {
  id: string;
  type: NotificationType;
  title?: string;
  message: string;
  duration?: number; // in milliseconds, 0 = doesn't auto-dismiss
  dismissible?: boolean;
  link?: { url: string; text: string };
  timestamp: Date;
  read?: boolean;
}

interface NotificationContextType {
  notifications: Notification[];
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => string;
  updateNotification: (id: string, notification: Partial<Notification>) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
  markAllAsRead: () => void;
  unreadCount: number;
}

// Create context
const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

// Provider component
export const NotificationProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [notificationContainer, setNotificationContainer] = useState<HTMLElement | null>(null);
  
  // Calculate unread count
  const unreadCount = notifications.filter(notification => !notification.read).length;

  // Set up notification container
  useEffect(() => {
    // Create container for notifications if it doesn't exist
    const container = document.getElementById('notification-container');
    if (container) {
      setNotificationContainer(container);
    } else {
      const newContainer = document.createElement('div');
      newContainer.id = 'notification-container';
      newContainer.className = 'fixed top-0 right-0 p-4 z-50 space-y-4 max-w-sm';
      document.body.appendChild(newContainer);
      setNotificationContainer(newContainer);
    }

    return () => {
      // Clean up on unmount
      const container = document.getElementById('notification-container');
      if (container && container.parentNode) {
        container.parentNode.removeChild(container);
      }
    };
  }, []);

  // Generate unique ID
  const generateId = (): string => {
    return Math.random().toString(36).substring(2, 11);
  };

  // Add a notification
  const addNotification = useCallback((notification: Omit<Notification, 'id' | 'timestamp'>): string => {
    const id = generateId();
    const newNotification: Notification = {
      ...notification,
      id,
      timestamp: new Date(),
      dismissible: notification.dismissible ?? true,
      duration: notification.duration ?? 5000, // Default 5 seconds
      read: false
    };

    setNotifications(prev => [newNotification, ...prev]);

    // Auto-dismiss after duration (if not 0)
    if (newNotification.duration && newNotification.duration > 0) {
      setTimeout(() => {
        removeNotification(id);
      }, newNotification.duration);
    }

    return id;
  }, []);

  // Update a notification
  const updateNotification = useCallback((id: string, notification: Partial<Notification>) => {
    setNotifications(prev =>
      prev.map(item =>
        item.id === id ? { ...item, ...notification } : item
      )
    );
  }, []);

  // Remove a notification
  const removeNotification = useCallback((id: string) => {
    setNotifications(prev =>
      prev.filter(notification => notification.id !== id)
    );
  }, []);

  // Clear all notifications
  const clearNotifications = useCallback(() => {
    setNotifications([]);
  }, []);
  
  // Mark all notifications as read
  const markAllAsRead = useCallback(() => {
    setNotifications(prev =>
      prev.map(notification => ({ ...notification, read: true }))
    );
  }, []);

  // Context value
  const value = {
    notifications,
    addNotification,
    updateNotification,
    removeNotification,
    clearNotifications,
    markAllAsRead,
    unreadCount
  };

  return (
    <NotificationContext.Provider value={value}>
      {children}
      {notificationContainer &&
        createPortal(<NotificationList />, notificationContainer)}
    </NotificationContext.Provider>
  );
};

// Hook to use notification context
export const useNotification = () => {
  const context = useContext(NotificationContext);
  if (context === undefined) {
    throw new Error('useNotification must be used within a NotificationProvider');
  }
  return context;
};

// Individual Notification Component
const NotificationItem: React.FC<{
  notification: Notification;
  onDismiss: () => void;
}> = ({ notification, onDismiss }) => {
  const { type, title, message, dismissible, link } = notification;
  const [isClosing, setIsClosing] = useState(false);

  useEffect(() => {
    // Entry animation happens automatically with CSS
    return () => {
      // Cleanup if component is removed during animation
    };
  }, []);

  const handleDismiss = () => {
    setIsClosing(true);
    // Delay removal to allow animation to complete
    setTimeout(() => {
      onDismiss();
    }, 300);
  };

  // Get icon based on notification type
  const getIcon = () => {
    switch (type) {
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'error':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'warning':
        return <AlertCircle className="w-5 h-5 text-yellow-500" />;
      case 'info':
      default:
        return <Info className="w-5 h-5 text-blue-500" />;
    }
  };

  // Get styles based on notification type
  const getStyles = () => {
    switch (type) {
      case 'success':
        return {
          container: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800',
          title: 'text-green-800 dark:text-green-300',
          message: 'text-green-700 dark:text-green-400'
        };
      case 'error':
        return {
          container: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800',
          title: 'text-red-800 dark:text-red-300',
          message: 'text-red-700 dark:text-red-400'
        };
      case 'warning':
        return {
          container: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800',
          title: 'text-yellow-800 dark:text-yellow-300',
          message: 'text-yellow-700 dark:text-yellow-400'
        };
      case 'info':
      default:
        return {
          container: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
          title: 'text-blue-800 dark:text-blue-300',
          message: 'text-blue-700 dark:text-blue-400'
        };
    }
  };

  const styles = getStyles();

  return (
    <div
      className={`
        ${styles.container} border rounded-lg shadow-lg p-4 
        transform transition-all duration-300 
        ${isClosing ? 'opacity-0 translate-x-full' : 'opacity-100'}
      `}
      role="alert"
    >
      <div className="flex">
        <div className="flex-shrink-0">
          {getIcon()}
        </div>
        <div className="ml-3 flex-1">
          {title && (
            <h3 className={`text-sm font-medium ${styles.title}`}>{title}</h3>
          )}
          <div className={`text-sm ${styles.message} ${title ? 'mt-1' : ''}`}>
            {message}
          </div>
          {link && (
            <div className="mt-2">
              <a
                href={link.url}
                className={`text-sm font-medium ${styles.title} hover:underline`}
                onClick={(e) => e.stopPropagation()}
                target="_blank"
                rel="noopener noreferrer"
              >
                {link.text}
              </a>
            </div>
          )}
        </div>
        {dismissible && (
          <button
            type="button"
            className={`ml-auto -mx-1.5 -my-1.5 ${styles.message} hover:bg-gray-100 dark:hover:bg-gray-800 rounded p-1.5`}
            onClick={handleDismiss}
            aria-label="Close"
          >
            <span className="sr-only">Close</span>
            <X className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
};

// Notification List Component
const NotificationList: React.FC = () => {
  const { notifications, removeNotification } = useNotification();

  if (notifications.length === 0) {
    return null;
  }

  return (
    <>
      {notifications.map((notification) => (
        <NotificationItem
          key={notification.id}
          notification={notification}
          onDismiss={() => removeNotification(notification.id)}
        />
      ))}
    </>
  );
};

// Notification Button Component for the header
export const NotificationButton: React.FC = () => {
  const { notifications, unreadCount, clearNotifications, markAllAsRead } = useNotification();
  const [isOpen, setIsOpen] = useState(false);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (isOpen && !(event.target as Element).closest('.notification-dropdown')) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  const toggleDropdown = () => {
    setIsOpen(!isOpen);
    if (!isOpen && unreadCount > 0) {
      // Mark as read when opening
      markAllAsRead();
    }
  };

  return (
    <div className="relative notification-dropdown">
      <button
        className="relative p-1 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 rounded-full"
        onClick={toggleDropdown}
        aria-label="Notifications"
      >
        <Bell size={20} />
        {unreadCount > 0 && (
          <span className="absolute top-0 right-0 inline-flex items-center justify-center px-2 py-1 text-xs font-bold leading-none text-red-100 bg-red-600 rounded-full">
            {unreadCount > 9 ? '9+' : unreadCount}
          </span>
        )}
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-80 bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden z-30 notification-dropdown-menu">
          {/* Header */}
          <div className="px-4 py-2 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
            <h3 className="text-sm font-medium text-gray-900 dark:text-white">Notifications</h3>
            {notifications.length > 0 && (
              <button
                className="text-xs text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-300"
                onClick={clearNotifications}
              >
                Clear all
              </button>
            )}
          </div>

          {/* Notification list */}
          <div className="max-h-96 overflow-y-auto">
            {notifications.length === 0 ? (
              <div className="px-4 py-6 text-center text-sm text-gray-500 dark:text-gray-400">
                No notifications
              </div>
            ) : (
              <ul>
                {notifications.map((notification) => (
                  <li
                    key={notification.id}
                    className={`
                      border-b border-gray-100 dark:border-gray-700 last:border-b-0
                      ${notification.read ? '' : 'bg-indigo-50 dark:bg-indigo-900/10'}
                    `}
                  >
                    <div className="px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-700 transition">
                      <div className="flex items-start">
                        <div className="flex-shrink-0 mt-0.5">
                          {notification.type === 'success' && <CheckCircle className="w-4 h-4 text-green-500" />}
                          {notification.type === 'error' && <XCircle className="w-4 h-4 text-red-500" />}
                          {notification.type === 'warning' && <AlertCircle className="w-4 h-4 text-yellow-500" />}
                          {notification.type === 'info' && <Info className="w-4 h-4 text-blue-500" />}
                        </div>
                        <div className="ml-2">
                          {notification.title && (
                            <p className="text-sm font-medium text-gray-900 dark:text-white">
                              {notification.title}
                            </p>
                          )}
                          <p className="text-xs text-gray-600 dark:text-gray-400 mt-0.5">
                            {notification.message}
                          </p>
                          <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                            {formatTimeAgo(notification.timestamp)}
                          </p>
                        </div>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* Footer */}
          <div className="px-4 py-2 border-t border-gray-200 dark:border-gray-700 text-center">
            <a
              href="#"
              className="text-xs text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-300"
              onClick={(e) => {
                e.preventDefault();
                setIsOpen(false);
              }}
            >
              View all notifications
            </a>
          </div>
        </div>
      )}
    </div>
  );
};

// Helper function to format time ago
const formatTimeAgo = (date: Date): string => {
  const now = new Date();
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);
  
  if (diffInSeconds < 60) {
    return 'just now';
  }
  
  const diffInMinutes = Math.floor(diffInSeconds / 60);
  if (diffInMinutes < 60) {
    return `${diffInMinutes} minute${diffInMinutes > 1 ? 's' : ''} ago`;
  }
  
  const diffInHours = Math.floor(diffInMinutes / 60);
  if (diffInHours < 24) {
    return `${diffInHours} hour${diffInHours > 1 ? 's' : ''} ago`;
  }
  
  const diffInDays = Math.floor(diffInHours / 24);
  if (diffInDays < 30) {
    return `${diffInDays} day${diffInDays > 1 ? 's' : ''} ago`;
  }
  
  const diffInMonths = Math.floor(diffInDays / 30);
  return `${diffInMonths} month${diffInMonths > 1 ? 's' : ''} ago`;
};

export default NotificationProvider;
