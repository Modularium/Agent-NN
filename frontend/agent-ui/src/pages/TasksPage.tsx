import { useEffect, useState } from 'react'

interface Task {
  id: string
  description: string
  status: string
  agent: string
}

export default function TasksPage() {
  const [tasks, setTasks] = useState<Task[]>([])

  useEffect(() => {
    fetch(`${import.meta.env.VITE_API_URL}/tasks`)
      .then(r => r.json())
      .then(setTasks)
      .catch(() => setTasks([]))
  }, [])

  return (
    <div className="p-4">
      <h2 className="text-xl mb-2">Tasks</h2>
      <table className="w-full text-left border-collapse">
        <thead>
          <tr className="border-b">
            <th className="p-2">Description</th>
            <th className="p-2">Status</th>
            <th className="p-2">Agent</th>
          </tr>
        </thead>
        <tbody>
          {tasks.map(t => (
            <tr key={t.id} className="border-b hover:bg-gray-100">
              <td className="p-2">{t.description}</td>
              <td className="p-2">{t.status}</td>
              <td className="p-2">{t.agent}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
