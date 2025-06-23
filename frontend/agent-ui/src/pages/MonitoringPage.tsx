import { useEffect, useState } from 'react'

export default function MonitoringPage() {
  const [metrics, setMetrics] = useState('')

  useEffect(() => {
    fetch(`${import.meta.env.VITE_API_URL}/metrics/summary`)
      .then(r => r.text())
      .then(setMetrics)
      .catch(() => setMetrics('error loading metrics'))
  }, [])

  return (
    <div className="p-4">
      <h2 className="text-xl mb-2">Monitoring</h2>
      <pre className="bg-gray-100 p-2 overflow-auto text-sm">
        {metrics}
      </pre>
    </div>
  )
}
