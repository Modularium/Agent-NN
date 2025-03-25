import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useTranslation } from '@smolitux/core';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ isOpen, onClose }) => {
  const location = useLocation();
  const t = useTranslation();

  const isActive = (path: string) => {
    return location.pathname === path ? 'active' : '';
  };

  return (
    <div className={`sidebar ${isOpen ? 'open' : ''}`}>
      <div className="sidebar-header">
        <h2>Agent-NN</h2>
        <button className="close-button" onClick={onClose}>×</button>
      </div>
      
      <nav className="sidebar-nav">
        <ul>
          <li className={isActive('/')}>
            <Link to="/" onClick={onClose}>
              <span className="nav-icon">🏠</span>
              <span className="nav-text">{t('navigation.home')}</span>
            </Link>
          </li>
          <li className={isActive('/chat')}>
            <Link to="/chat" onClick={onClose}>
              <span className="nav-icon">💬</span>
              <span className="nav-text">{t('navigation.chat')}</span>
            </Link>
          </li>
          <li className={isActive('/agents')}>
            <Link to="/agents" onClick={onClose}>
              <span className="nav-icon">🤖</span>
              <span className="nav-text">{t('navigation.agents')}</span>
            </Link>
          </li>
          <li className={isActive('/tasks')}>
            <Link to="/tasks" onClick={onClose}>
              <span className="nav-icon">📋</span>
              <span className="nav-text">{t('navigation.tasks')}</span>
            </Link>
          </li>
          <li className={isActive('/settings')}>
            <Link to="/settings" onClick={onClose}>
              <span className="nav-icon">⚙️</span>
              <span className="nav-text">{t('navigation.settings')}</span>
            </Link>
          </li>
        </ul>
      </nav>
      
      <div className="sidebar-footer">
        <p>© {new Date().getFullYear()} EcoSphereNetwork</p>
      </div>
    </div>
  );
};

export default Sidebar;