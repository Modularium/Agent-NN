import { useState } from 'react'

export default function FeedbackPage() {
  const [feedback, setFeedback] = useState('')
  const [sessionId] = useState(() => localStorage.getItem('sessionId') || '')

  const submit = async () => {
    await fetch(`${import.meta.env.VITE_API_URL}/session/${sessionId}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: feedback }),
    })
    setFeedback('')
  }

  return (
    <div className="p-4">
      <h2 className="text-xl mb-2">Feedback</h2>
      <textarea
        className="border w-full p-2 mb-2"
        value={feedback}
        onChange={e => setFeedback(e.target.value)}
        placeholder="Enter feedback"
      />
      <button className="bg-blue-600 text-white px-4" onClick={submit}>
        Submit
      </button>
    </div>
  )
}
