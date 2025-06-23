import { useState } from 'react'

export default function SettingsPage() {
  const [apiKey, setApiKey] = useState('')
  const [model, setModel] = useState('openai')

  const save = () => {
    localStorage.setItem('apiKey', apiKey)
    localStorage.setItem('model', model)
  }

  return (
    <div className="p-4 space-y-2">
      <h2 className="text-xl">Settings</h2>
      <div>
        <label className="block mb-1">API Key</label>
        <input
          className="border p-2 w-full"
          type="password"
          value={apiKey}
          onChange={e => setApiKey(e.target.value)}
        />
      </div>
      <div>
        <label className="block mb-1">Model</label>
        <select
          className="border p-2 w-full"
          value={model}
          onChange={e => setModel(e.target.value)}
        >
          <option value="openai">OpenAI</option>
          <option value="local">Local</option>
        </select>
      </div>
      <button className="bg-blue-600 text-white px-4" onClick={save}>
        Save
      </button>
    </div>
  )
}
