import { NavLink } from 'react-router-dom'

export default function Sidebar() {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `block px-4 py-2 hover:bg-gray-700 ${isActive ? 'bg-gray-700' : ''}`

  return (
    <nav className="w-48 bg-gray-800 text-white min-h-screen">
      <ul>
        <li>
          <NavLink to="/" end className={linkClass}>
            Chat
          </NavLink>
        </li>
        <li>
          <NavLink to="/routing" className={linkClass}>
            Routing
          </NavLink>
        </li>
        <li>
          <NavLink to="/feedback" className={linkClass}>
            Feedback
          </NavLink>
        </li>
        <li>
          <NavLink to="/monitoring" className={linkClass}>
            Monitoring
          </NavLink>
        </li>
      </ul>
    </nav>
  )
}
