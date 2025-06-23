import { useState } from 'react'

export default function FeedbackPage() {
  const [feedback, setFeedback] = useState('')
  const [sessionId] = useState(() => localStorage.getItem('sessionId') || '')
  const [error, setError] = useState('')
  const [success, setSuccess] = useState(false)

  const submit = async () => {
    setSuccess(false)
    setError('')
    if (!feedback.trim()) {
      setError('Feedback darf nicht leer sein')
      return
    }
    try {
      const res = await fetch(`${import.meta.env.VITE_API_URL}/session/${sessionId}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: feedback }),
      })
      if (!res.ok) throw new Error('Request failed')
      setSuccess(true)
      setFeedback('')
    } catch {
      setError('Fehler beim Senden')
    }
  }

  return (
    <div className="p-4">
      <h2 className="text-xl mb-2">Feedback</h2>
      {error && <div className="text-red-600" role="alert">{error}</div>}
      {success && <div className="text-green-700" role="status">Danke für dein Feedback!</div>}
      <textarea
        className="border w-full p-2 mb-2"
        value={feedback}
        onChange={e => setFeedback(e.target.value)}
        placeholder="Enter feedback"
        aria-label="Feedback"
      />
      <button className="bg-blue-600 text-white px-4" onClick={submit} aria-label="Submit feedback">
        Submit
      </button>
    </div>
  )
}
