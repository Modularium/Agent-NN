import React, { useState, useEffect } from 'react'
import { 
  Card, 
  Input, 
  Button, 
  Alert,
  Spinner,
  useTranslation
} from '@smolitux/core'

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
      // In a real implementation, this would be an API call
      // For demo purposes, we'll simulate a response
      setTimeout(() => {
        const agentResponse: Message = {
          role: 'assistant',
          content: `This is a simulated response to: "${input}"`,
          metadata: {
            agent: 'finance_agent',
            executionTime: 1.2
          }
        }
        setMessages(prev => [...prev, agentResponse])
        setLoading(false)
      }, 1500)
      
      // Real API call would look like this:
      /*
      const response = await fetch('/api/tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_description: input })
      })
      
      const data = await response.json()
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.result,
        metadata: {
          agent: data.chosen_agent,
          executionTime: data.execution_time
        }
      }])
      */
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