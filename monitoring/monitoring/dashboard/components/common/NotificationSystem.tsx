// monitoring/dashboard/components/common/NotificationSystem.tsx
import React, { useState, useEffect, useCallback, createContext, useContext } from 'react';
import { AlertCircle, CheckCircle, Info, X, XCircle, Bell } from 'lucide-react';
import { createPortal } from 'react-dom';

// Types
export type NotificationType = 'success' | 'error' | 'warning' | 'info';

export interface Notification {
  id: string;
  type: NotificationType;
  title?: string;
  message: string;
  duration?: number;
  dismissible?: boolean;
  link?: { url: string; text: string };
  onClick?: () => void;
}

interface NotificationContextType {
  notifications: Notification[];
  addNotification: (notification: Omit<Notification, 'id'>) => string;
  updateNotification: (id: string, notification: Partial<Notification>) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
}

// Context
const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

// Provider Component
export const NotificationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [notificationContainer, setNotificationContainer] = useState<HTMLElement | null>(null);

  useEffect(() => {
    // Create container for notifications
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
  const generateId = () => {
    return Math.random().toString(36).substring(2, 11);
  };

  // Add a notification
  const addNotification = useCallback((notification: Omit<Notification, 'id'>) => {
    const id = generateId();
    const newNotification: Notification = {
      ...notification,
      id,
      dismissible: notification.dismissible ?? true,
      duration: notification.duration ?? 5000, // Default 5 seconds
    };

    setNotifications((prevNotifications) => [
      ...prevNotifications,
      newNotification,
    ]);

    // Auto-dismiss after duration (if not -1)
    if (newNotification.duration && newNotification.duration > 0) {
      setTimeout(() => {
        removeNotification(id);
      }, newNotification.duration);
    }

    return id;
  }, []);

  // Update a notification
  const updateNotification = useCallback((id: string, notification: Partial<Notification>) => {
    setNotifications((prevNotifications) =>
      prevNotifications.map((item) =>
        item.id === id ? { ...item, ...notification } : item
      )
    );
  }, []);

  // Remove a notification
  const removeNotification = useCallback((id: string) => {
    setNotifications((prevNotifications) =>
      prevNotifications.filter((notification) => notification.id !== id)
    );
  }, []);

  // Clear all notifications
  const clearNotifications = useCallback(() => {
    setNotifications([]);
  }, []);

  const contextValue = {
    notifications,
    addNotification,
    updateNotification,
    removeNotification,
    clearNotifications,
  };

  return (
    <NotificationContext.Provider value={contextValue}>
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
  const { type, title, message, dismissible, link, onClick } = notification;
  const [isClosing, setIsClosing] = useState(false);

  // Animations for entry and exit
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

  const handleClick = () => {
    if (onClick) {
      onClick();
    }
  };

  // Icon based on notification type
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

  // Background color based on notification type
  const getBackgroundColor = () => {
    switch (type) {
      case 'success':
        return 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800';
      case 'error':
        return 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800';
      case 'warning':
        return 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800';
      case 'info':
      default:
        return 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800';
    }
  };

  // Text color based on notification type
  const getTextColor = () => {
    switch (type) {
      case 'success':
        return 'text-green-800 dark:text-green-300';
      case 'error':
        return 'text-red-800 dark:text-red-300';
      case 'warning':
        return 'text-yellow-800 dark:text-yellow-300';
      case 'info':
      default:
        return 'text-blue-800 dark:text-blue-300';
    }
  };

  return (
    <div
      className={`${getBackgroundColor()} border rounded-lg shadow-lg p-4 transform transition-all duration-300 ${
        isClosing ? 'opacity-0 translate-x-full' : 'opacity-100'
      } ${onClick ? 'cursor-pointer' : ''}`}
      onClick={handleClick}
      role="alert"
    >
      <div className="flex">
        <div className="flex-shrink-0">{getIcon()}</div>
        <div className="ml-3 flex-1">
          {title && (
            <h3 className={`text-sm font-medium ${getTextColor()}`}>
              {title}
            </h3>
          )}
          <div className={`text-sm ${title ? 'mt-1' : ''} ${getTextColor()}`}>
            {message}
          </div>
          {link && (
            <div className="mt-2">
              <a
                href={link.url}
                className={`text-sm font-medium ${getTextColor()} hover:underline`}
                onClick={(e) => e.stopPropagation()}
              >
                {link.text}
              </a>
            </div>
          )}
        </div>
        {dismissible && (
          <div className="ml-4 flex-shrink-0 flex">
            <button
              type="button"
              className={`bg-transparent rounded-md inline-flex ${getTextColor()} hover:bg-gray-200 dark:hover:bg-gray-700 focus:outline-none`}
              onClick={(e) => {
                e.stopPropagation();
                handleDismiss();
              }}
            >
              <span className="sr-only">Close</span>
              <X className="h-5 w-5" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

// List of all notifications
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

// Notification button component to show/hide notifications panel
export const NotificationButton: React.FC = () => {
  const { notifications, clearNotifications } = useNotification();
  const [showPanel, setShowPanel] = useState(false);
  const hasUnread = notifications.length > 0;

  const togglePanel = () => {
    setShowPanel(!showPanel);
  };

  return (
    <div className="relative">
      <button
        className="relative p-1 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 rounded"
        onClick={togglePanel}
        aria-label="Notifications"
      >
        <Bell size={20} />
        {hasUnread && (
          <span className="absolute top-0 right-0 block h-2 w-2 rounded-full bg-red-500"></span>
        )}
      </button>

      {showPanel && (
        <div className="origin-top-right absolute right-0 mt-2 w-80 rounded-md shadow-lg bg-white dark:bg-gray-800 ring-1 ring-black ring-opacity-5 z-50">
          <div className="py-1">
            <div className="px-4 py-2 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
              <h3 className="text-sm font-medium text-gray-900 dark:text-white">Notifications</h3>
              {notifications.length > 0 && (
                <button
                  className="text-sm text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-300"
                  onClick={clearNotifications}
                >
                  Clear all
                </button>
              )}
            </div>
            <div className="max-h-60 overflow-y-auto">
              {notifications.length === 0 ? (
                <div className="px-4 py-2 text-sm text-gray-500 dark:text-gray-400">
                  No notifications
                </div>
              ) : (
                notifications.map((notification) => (
                  <div
                    key={notification.id}
                    className="block px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 border-b border-gray-100 dark:border-gray-800"
                  >
                    <div className="flex items-start">
                      <div className="flex-shrink-0 mt-0.5">
                        {notification.type === 'success' && <CheckCircle className="w-4 h-4 text-green-500" />}
                        {notification.type === 'error' && <XCircle className="w-4 h-4 text-red-500" />}
                        {notification.type === 'warning' && <AlertCircle className="w-4 h-4 text-yellow-500" />}
                        {notification.type === 'info' && <Info className="w-4 h-4 text-blue-500" />}
                      </div>
                      <div className="ml-2">
                        {notification.title && (
                          <div className="font-medium">{notification.title}</div>
                        )}
                        <div className="text-xs mt-0.5">{notification.message}</div>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
            <div className="border-t border-gray-200 dark:border-gray-700">
              <a 
                href="#" 
                className="block px-4 py-2 text-sm text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-300"
                onClick={(e) => {
                  e.preventDefault();
                  setShowPanel(false);
                }}
              >
                View all notifications
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Notification examples for development
export const NotificationExample: React.FC = () => {
  const { addNotification } = useNotification();

  const showSuccessNotification = () => {
    addNotification({
      type: 'success',
      title: 'Operation Successful',
      message: 'Your changes have been saved successfully.',
      duration: 5000,
    });
  };

  const showErrorNotification = () => {
    addNotification({
      type: 'error',
      title: 'Error Occurred',
      message: 'There was an error processing your request. Please try again.',
      duration: 0, // Won't auto-dismiss
    });
  };

  const showWarningNotification = () => {
    addNotification({
      type: 'warning',
      title: 'Warning',
      message: 'This action cannot be undone. Please proceed with caution.',
      duration: 7000,
    });
  };

  const showInfoNotification = () => {
    addNotification({
      type: 'info',
      title: 'Information',
      message: 'The system will be undergoing maintenance in 30 minutes.',
      link: {
        url: '#',
        text: 'Learn more',
      },
    });
  };

  return (
    <div className="space-y-2">
      <button
        className="px-4 py-2 bg-green-600 text-white rounded-md"
        onClick={showSuccessNotification}
      >
        Show Success
      </button>
      <button
        className="px-4 py-2 bg-red-600 text-white rounded-md"
        onClick={showErrorNotification}
      >
        Show Error
      </button>
      <button
        className="px-4 py-2 bg-yellow-600 text-white rounded-md"
        onClick={showWarningNotification}
      >
        Show Warning
      </button>
      <button
        className="px-4 py-2 bg-blue-600 text-white rounded-md"
        onClick={showInfoNotification}
      >
        Show Info
      </button>
    </div>
  );
};
