import { NavLink } from 'react-router-dom'
import type { Dispatch, SetStateAction } from 'react'
interface Props {
  open: boolean
  setOpen: Dispatch<SetStateAction<boolean>>
}

export default function Sidebar({ open, setOpen }: Props) {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `block px-4 py-2 hover:bg-gray-700 ${isActive ? 'bg-gray-700' : ''}`

  return (
    <nav
      id="sidebar"
      className={`${open ? 'block' : 'hidden'} sm:block w-48 bg-gray-800 text-white h-full fixed sm:static z-20`}
      aria-label="Main navigation"
    >
      <button
        className="sm:hidden p-2 text-right w-full"
        onClick={() => setOpen(false)}
        aria-label="Close menu"
      >
        ✕
      </button>
      <ul className="mt-2">
        <li>
          <NavLink to="/" end className={linkClass} onClick={() => setOpen(false)}>
            Chat
          </NavLink>
        </li>
        <li>
          <NavLink to="/agents" className={linkClass} onClick={() => setOpen(false)}>
            Agents
          </NavLink>
        </li>
        <li>
          <NavLink to="/tasks" className={linkClass} onClick={() => setOpen(false)}>
            Tasks
          </NavLink>
        </li>
        <li>
          <NavLink to="/routing" className={linkClass} onClick={() => setOpen(false)}>
            Routing
          </NavLink>
        </li>
        <li>
          <NavLink to="/feedback" className={linkClass} onClick={() => setOpen(false)}>
            Feedback
          </NavLink>
        </li>
        <li>
          <NavLink to="/monitoring" className={linkClass} onClick={() => setOpen(false)}>
            Monitoring
          </NavLink>
        </li>
        <li>
          <NavLink to="/metrics" className={linkClass} onClick={() => setOpen(false)}>
            Metrics
          </NavLink>
        </li>
        <li>
          <NavLink to="/settings" className={linkClass} onClick={() => setOpen(false)}>
            Settings
          </NavLink>
        </li>
        <li>
          <NavLink to="/admin" className={linkClass} onClick={() => setOpen(false)}>
            Admin
          </NavLink>
        </li>
        <li>
          <NavLink to="/debug" className={linkClass} onClick={() => setOpen(false)}>
            Debug
          </NavLink>
        </li>
      </ul>
    </nav>
  )
}
