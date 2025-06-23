import { useState } from 'react'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

export default function ChatPage() {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [taskType, setTaskType] = useState('default')
  const [sessionId] = useState(() => localStorage.getItem('sessionId') || '')

  const sendMessage = async () => {
    if (!input.trim()) return
    const userMessage: Message = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    const res = await fetch(`${import.meta.env.VITE_API_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: input, task_type: taskType, session_id: sessionId }),
    })
    const data = await res.json()
    const assistantMessage: Message = { role: 'assistant', content: data.text || data.message }
    setMessages(prev => [...prev, assistantMessage])
  }

  return (
    <div className="p-4 flex flex-col h-full">
      <div className="flex-1 overflow-y-auto mb-4">
        {messages.map((m, idx) => (
          <div key={idx} className={m.role === 'user' ? 'text-right' : 'text-left'}>
            <div className="inline-block px-3 py-2 my-1 rounded bg-gray-200 text-gray-800 max-w-xl">
              {m.content}
            </div>
          </div>
        ))}
      </div>
      <div className="flex gap-2">
        <select value={taskType} onChange={e => setTaskType(e.target.value)} className="border p-2">
          <option value="default">default</option>
          <option value="docker">docker</option>
          <option value="container_ops">container_ops</option>
        </select>
        <input
          className="flex-1 border p-2"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Enter prompt..."
        />
        <button className="bg-blue-600 text-white px-4" onClick={sendMessage}>
          Send
        </button>
      </div>
    </div>
  )
}
