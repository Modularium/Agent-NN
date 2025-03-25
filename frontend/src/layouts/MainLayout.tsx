import { Outlet, Link, useLocation } from 'react-router-dom'
import { 
  Card, 
  useTranslation,
  Button,
  Drawer,
  Switch
} from '@smolitux/core'
import { useState } from 'react'

const MainLayout = () => {
  const [darkMode, setDarkMode] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)
  const location = useLocation()
  const t = useTranslation()

  const toggleDarkMode = () => {
    setDarkMode(!darkMode)
    // In a real app, this would update the theme
    document.body.classList.toggle('dark-mode')
  }

  const isActive = (path: string) => {
    return location.pathname === path ? 'active' : ''
  }

  return (
    <div className={`app-container ${darkMode ? 'dark-mode' : ''}`}>
      <header className="app-header">
        <div className="container">
          <div className="header-content">
            <div className="logo">
              <h1>Agent-NN</h1>
            </div>
            <div className="header-actions">
              <Button 
                variant="text" 
                onClick={() => setMenuOpen(true)}
                className="menu-button"
              >
                Menu
              </Button>
              <div className="theme-toggle">
                <span>{t('theme.light')}</span>
                <Switch 
                  checked={darkMode} 
                  onChange={toggleDarkMode} 
                />
                <span>{t('theme.dark')}</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <Drawer
        open={menuOpen}
        onClose={() => setMenuOpen(false)}
        position="left"
      >
        <div className="drawer-content">
          <h2>{t('navigation.menu')}</h2>
          <nav className="drawer-nav">
            <ul>
              <li className={isActive('/')}>
                <Link to="/" onClick={() => setMenuOpen(false)}>
                  {t('navigation.home')}
                </Link>
              </li>
              <li className={isActive('/chat')}>
                <Link to="/chat" onClick={() => setMenuOpen(false)}>
                  {t('navigation.chat')}
                </Link>
              </li>
              <li className={isActive('/agents')}>
                <Link to="/agents" onClick={() => setMenuOpen(false)}>
                  {t('navigation.agents')}
                </Link>
              </li>
              <li className={isActive('/tasks')}>
                <Link to="/tasks" onClick={() => setMenuOpen(false)}>
                  {t('navigation.tasks')}
                </Link>
              </li>
              <li className={isActive('/settings')}>
                <Link to="/settings" onClick={() => setMenuOpen(false)}>
                  {t('navigation.settings')}
                </Link>
              </li>
            </ul>
          </nav>
        </div>
      </Drawer>

      <main className="app-main">
        <div className="container">
          <Outlet />
        </div>
      </main>

      <footer className="app-footer">
        <div className="container">
          <p>&copy; {new Date().getFullYear()} EcoSphereNetwork - Agent-NN</p>
        </div>
      </footer>
    </div>
  )
}

export default MainLayout