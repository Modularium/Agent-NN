import { Outlet } from 'react-router-dom'
import { useState } from 'react'
import Navbar from '../components/Navbar'
import Sidebar from '../components/Sidebar'

const MainLayout = () => {
  const [menuOpen, setMenuOpen] = useState(false)

  return (
    <div className="app-container">
      <Navbar onMenuClick={() => setMenuOpen(true)} />
      
      <Sidebar 
        isOpen={menuOpen} 
        onClose={() => setMenuOpen(false)} 
      />

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