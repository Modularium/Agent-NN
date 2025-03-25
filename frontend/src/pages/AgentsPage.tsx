import React, { useState, useEffect } from 'react'
import { 
  Card, 
  Table, 
  Badge, 
  Button,
  Modal,
  Spinner,
  useTranslation
} from '@smolitux/core'

interface Agent {
  id: string
  name: string
  domain: string
  totalTasks: number
  successRate: number
  description: string
  avgExecutionTime: number
  knowledgeBase: {
    documentsCount: number
  }
}

const AgentsPage: React.FC = () => {
  const [agents, setAgents] = useState<Agent[]>([])
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null)
  const [showModal, setShowModal] = useState(false)
  const [loading, setLoading] = useState(true)
  const t = useTranslation()

  useEffect(() => {
    // Simulate loading agents data
    setTimeout(() => {
      const mockAgents: Agent[] = [
        {
          id: '1',
          name: 'finance_agent',
          domain: 'Finance',
          totalTasks: 124,
          successRate: 0.92,
          description: 'Specialized in financial analysis and reporting',
          avgExecutionTime: 1.8,
          knowledgeBase: {
            documentsCount: 342
          }
        },
        {
          id: '2',
          name: 'tech_agent',
          domain: 'Technology',
          totalTasks: 98,
          successRate: 0.85,
          description: 'Expert in software development and technical solutions',
          avgExecutionTime: 2.1,
          knowledgeBase: {
            documentsCount: 287
          }
        },
        {
          id: '3',
          name: 'marketing_agent',
          domain: 'Marketing',
          totalTasks: 76,
          successRate: 0.78,
          description: 'Specialized in marketing strategy and analysis',
          avgExecutionTime: 1.5,
          knowledgeBase: {
            documentsCount: 215
          }
        }
      ]
      setAgents(mockAgents)
      setLoading(false)
    }, 1000)
    
    // In a real implementation, this would be an API call:
    /*
    const fetchAgents = async () => {
      try {
        const response = await fetch('/api/agents')
        const data = await response.json()
        setAgents(data)
      } catch (error) {
        console.error('Failed to fetch agents:', error)
      } finally {
        setLoading(false)
      }
    }
    
    fetchAgents()
    */
  }, [])

  const handleAgentClick = (agent: Agent) => {
    setSelectedAgent(agent)
    setShowModal(true)
  }

  const getBadgeType = (rate: number) => {
    if (rate > 0.8) return 'success'
    if (rate > 0.5) return 'warning'
    return 'danger'
  }

  return (
    <div className="agents-page">
      <h1>{t('agents.title')}</h1>
      
      <Card className="agents-container">
        {loading ? (
          <div className="loading-container">
            <Spinner size="lg" />
            <p>{t('agents.loading')}</p>
          </div>
        ) : (
          <Table>
            <thead>
              <tr>
                <th>{t('agents.name')}</th>
                <th>{t('agents.domain')}</th>
                <th>{t('agents.tasks')}</th>
                <th>{t('agents.successRate')}</th>
                <th>{t('agents.actions')}</th>
              </tr>
            </thead>
            <tbody>
              {agents.map(agent => (
                <tr key={agent.id}>
                  <td>{agent.name}</td>
                  <td>{agent.domain}</td>
                  <td>{agent.totalTasks}</td>
                  <td>
                    <Badge 
                      type={getBadgeType(agent.successRate)}
                    >
                      {(agent.successRate * 100).toFixed(1)}%
                    </Badge>
                  </td>
                  <td>
                    <Button 
                      variant="secondary" 
                      size="sm"
                      onClick={() => handleAgentClick(agent)}
                    >
                      {t('agents.details')}
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </Card>

      {selectedAgent && (
        <Modal
          title={`${t('agents.details')}: ${selectedAgent.name}`}
          open={showModal}
          onClose={() => setShowModal(false)}
        >
          <div className="agent-details">
            <h3>{t('agents.domain')}: {selectedAgent.domain}</h3>
            <p>{t('agents.description')}: {selectedAgent.description}</p>
            
            <h4>{t('agents.performance')}</h4>
            <ul>
              <li>{t('agents.totalTasks')}: {selectedAgent.totalTasks}</li>
              <li>{t('agents.successRate')}: {(selectedAgent.successRate * 100).toFixed(1)}%</li>
              <li>{t('agents.avgExecutionTime')}: {selectedAgent.avgExecutionTime.toFixed(2)}s</li>
            </ul>
            
            <h4>{t('agents.knowledgeBase')}</h4>
            <p>{t('agents.documentsCount')}: {selectedAgent.knowledgeBase.documentsCount}</p>
          </div>
        </Modal>
      )}
    </div>
  )
}

export default AgentsPage