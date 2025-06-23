import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Button, useTranslation } from '@smolitux/core';
import { useTheme } from '../utils/theme';

interface NavbarProps {
  onMenuClick: () => void;
}

const Navbar: React.FC<NavbarProps> = ({ onMenuClick }) => {
  const location = useLocation();
  const t = useTranslation();
  const { theme, setTheme } = useTheme();

  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };

  const getTitle = () => {
    switch (location.pathname) {
      case '/':
        return t('home.title');
      case '/chat':
        return t('chat.title');
      case '/agents':
        return t('agents.title');
      case '/tasks':
        return t('tasks.title');
      case '/settings':
        return t('settings.title');
      default:
        return 'Agent-NN';
    }
  };

  return (
    <header className="navbar">
      <div className="navbar-container">
        <div className="navbar-left">
          <Button 
            variant="text" 
            onClick={onMenuClick}
            className="menu-button"
          >
            <span className="menu-icon">☰</span>
          </Button>
          <h1 className="navbar-title">{getTitle()}</h1>
        </div>
        
        <div className="navbar-right">
          <Button 
            variant="text" 
            onClick={toggleTheme}
            className="theme-toggle"
          >
            {theme === 'light' ? '🌙' : '☀️'}
          </Button>
          
          <Link to="/chat" className="navbar-action">
            <Button variant="primary" size="sm">
              {t('chat.title')}
            </Button>
          </Link>
        </div>
      </div>
    </header>
  );
};

export default Navbar;