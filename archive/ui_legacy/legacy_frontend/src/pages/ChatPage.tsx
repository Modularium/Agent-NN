import React, { useState, useEffect } from 'react'
import {
  Card,
  Input,
  Button,
  Alert,
  Spinner,
  Select,
  useTranslation
} from '@smolitux/core'
import apiClient from '../utils/api'

interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
  error?: boolean
  metadata?: {
    agent?: string
    executionTime?: number
  }
}

const ChatPage: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [agent, setAgent] = useState('dev')
  const [sessionId, setSessionId] = useState<string | null>(null)
  const t = useTranslation()

  // Scroll to bottom when messages change
  useEffect(() => {
    const messagesContainer = document.querySelector('.messages-container')
    if (messagesContainer) {
      messagesContainer.scrollTop = messagesContainer.scrollHeight
    }
  }, [messages])

  const sendMessage = async () => {
    if (!input.trim()) return
    
    // Add user message
    const userMessage: Message = { role: 'user', content: input }
    setMessages([...messages, userMessage])
    setInput('')
    setLoading(true)
    
    try {
      const response = await apiClient.sendChatMessage(input, agent, sessionId || undefined)
      setSessionId(response.session_id)
      const agentResponse: Message = {
        role: 'assistant',
        content: response.response.result ?? response.response,
        metadata: {
          agent: response.worker,
          executionTime: response.duration
        }
      }
      setMessages(prev => [...prev, agentResponse])
      setLoading(false)
    } catch (error) {
      // Error handling
      setMessages(prev => [...prev, {
        role: 'system',
        content: t('errors.requestFailed'),
        error: true
      }])
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="chat-page">
      <h1>{t('chat.title')}</h1>
      
      <Card className="chat-container">
        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="empty-chat">
              <p>{t('chat.empty')}</p>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div key={index} className={`message ${msg.role}`}>
                {msg.error ? (
                  <Alert type="error">{msg.content}</Alert>
                ) : (
                  <>
                    <div className="message-content">{msg.content}</div>
                    {msg.metadata && (
                      <div className="message-metadata">
                        {msg.metadata.agent && (
                          <span>{t('chat.processedBy')}: {msg.metadata.agent}</span>
                        )}
                        {msg.metadata.executionTime && (
                          <span> ({msg.metadata.executionTime.toFixed(2)}s)</span>
                        )}
                      </div>
                    )}
                  </>
                )}
              </div>
            ))
          )}
          {loading && (
            <div className="message system">
              <Spinner size="sm" /> {t('chat.thinking')}
            </div>
          )}
        </div>
        
        <div className="input-container">
          <Select value={agent} onChange={(v) => setAgent(v as string)}>
            <option value="dev">dev</option>
            <option value="openhands">openhands</option>
            <option value="plugin-agent">plugin-agent</option>
          </Select>
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={t('chat.inputPlaceholder')}
            disabled={loading}
          />
          <Button 
            variant="primary" 
            onClick={sendMessage}
            disabled={loading || !input.trim()}
          >
            {t('chat.send')}
          </Button>
        </div>
      </Card>
    </div>
  )
}

export default ChatPage
