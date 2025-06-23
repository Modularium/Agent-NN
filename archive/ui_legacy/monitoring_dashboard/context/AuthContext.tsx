// AuthContext.tsx - Authentication context for the dashboard
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

// Define types for user and auth context
interface User {
  username: string;
  role: string;
  permissions: string[];
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  loading: boolean;
  error: string | null;
}

// Create the context with undefined as default value
const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Provider component
export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Check for existing auth token on mount
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const token = localStorage.getItem('authToken');
        if (token) {
          // In a real app, validate the token with an API
          // For now, we'll just assume it's valid and set some user data
          setUser({
            username: 'admin',
            role: 'administrator',
            permissions: ['read', 'write', 'manage']
          });
        }
      } catch (err) {
        console.error('Auth validation error:', err);
      } finally {
        setLoading(false);
      }
    };
    
    checkAuth();
  }, []);

  // Login function
  const login = async (username: string, password: string) => {
    setLoading(true);
    setError(null);
    
    try {
      // In a real app, make an API call to your auth server
      // For this demo, we'll simulate a login
      if (username === 'admin' && password === 'password') {
        const user = {
          username: 'admin',
          role: 'administrator',
          permissions: ['read', 'write', 'manage']
        };
        
        localStorage.setItem('authToken', 'mock-token');
        localStorage.setItem('user', JSON.stringify(user));
        setUser(user);
      } else {
        throw new Error('Invalid credentials');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Login failed';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  // Logout function
  const logout = () => {
    localStorage.removeItem('authToken');
    localStorage.removeItem('user');
    setUser(null);
  };

  // Context value
  const contextValue: AuthContextType = {
    user,
    isAuthenticated: !!user,
    login,
    logout,
    loading,
    error
  };

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
};

// Custom hook to use the auth context
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
