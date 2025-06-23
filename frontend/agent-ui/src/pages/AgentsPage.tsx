import { useEffect, useState } from 'react'

interface Agent {
  id: string
  name: string
  domain: string
  totalTasks: number
  successRate: number
}

export default function AgentsPage() {
  const [agents, setAgents] = useState<Agent[]>([])

  useEffect(() => {
    fetch(`${import.meta.env.VITE_API_URL}/agents`)
      .then(r => r.json())
      .then(setAgents)
      .catch(() => setAgents([]))
  }, [])

  return (
    <div className="p-4">
      <h2 className="text-xl mb-2">Agents</h2>
      <table className="w-full text-left border-collapse">
        <thead>
          <tr className="border-b">
            <th className="p-2">Name</th>
            <th className="p-2">Domain</th>
            <th className="p-2">Tasks</th>
            <th className="p-2">Success</th>
          </tr>
        </thead>
        <tbody>
          {agents.map(a => (
            <tr key={a.id} className="border-b hover:bg-gray-100">
              <td className="p-2">{a.name}</td>
              <td className="p-2">{a.domain}</td>
              <td className="p-2">{a.totalTasks}</td>
              <td className="p-2">{(a.successRate * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
