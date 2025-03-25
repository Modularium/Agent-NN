import React, { useState, useEffect } from 'react'
import { 
  Card, 
  Table, 
  Badge, 
  Button,
  Timeline,
  Spinner,
  useTranslation
} from '@smolitux/core'

interface TaskEvent {
  type: string
  timestamp: string
  description: string
  status: 'success' | 'error' | 'warning' | 'info'
  metadata?: {
    agent?: string
    executionTime?: number
  }
}

interface Task {
  id: string
  description: string
  status: 'completed' | 'in_progress' | 'failed' | 'pending'
  timestamp: string
  agent: string
  executionTime: number
  events: TaskEvent[]
  result?: string
}

const TasksPage: React.FC = () => {
  const [tasks, setTasks] = useState<Task[]>([])
  const [selectedTask, setSelectedTask] = useState<Task | null>(null)
  const [loading, setLoading] = useState(true)
  const t = useTranslation()

  useEffect(() => {
    // Simulate loading tasks data
    setTimeout(() => {
      const mockTasks: Task[] = [
        {
          id: '1',
          description: 'Analyze quarterly financial report',
          status: 'completed',
          timestamp: new Date(Date.now() - 3600000).toISOString(),
          agent: 'finance_agent',
          executionTime: 2.3,
          events: [
            {
              type: 'task_received',
              timestamp: new Date(Date.now() - 3660000).toISOString(),
              description: 'Task received by system',
              status: 'info'
            },
            {
              type: 'agent_selection',
              timestamp: new Date(Date.now() - 3650000).toISOString(),
              description: 'Agent selected for task',
              status: 'info',
              metadata: {
                agent: 'finance_agent'
              }
            },
            {
              type: 'execution',
              timestamp: new Date(Date.now() - 3630000).toISOString(),
              description: 'Task executed successfully',
              status: 'success',
              metadata: {
                executionTime: 2.3
              }
            }
          ],
          result: 'Financial report analysis complete. Revenue increased by 12% compared to previous quarter.'
        },
        {
          id: '2',
          description: 'Research new cloud technologies',
          status: 'in_progress',
          timestamp: new Date(Date.now() - 1800000).toISOString(),
          agent: 'tech_agent',
          executionTime: 0,
          events: [
            {
              type: 'task_received',
              timestamp: new Date(Date.now() - 1860000).toISOString(),
              description: 'Task received by system',
              status: 'info'
            },
            {
              type: 'agent_selection',
              timestamp: new Date(Date.now() - 1850000).toISOString(),
              description: 'Agent selected for task',
              status: 'info',
              metadata: {
                agent: 'tech_agent'
              }
            },
            {
              type: 'execution',
              timestamp: new Date(Date.now() - 1840000).toISOString(),
              description: 'Task execution in progress',
              status: 'warning'
            }
          ]
        },
        {
          id: '3',
          description: 'Optimize marketing campaign',
          status: 'failed',
          timestamp: new Date(Date.now() - 7200000).toISOString(),
          agent: 'marketing_agent',
          executionTime: 1.5,
          events: [
            {
              type: 'task_received',
              timestamp: new Date(Date.now() - 7260000).toISOString(),
              description: 'Task received by system',
              status: 'info'
            },
            {
              type: 'agent_selection',
              timestamp: new Date(Date.now() - 7250000).toISOString(),
              description: 'Agent selected for task',
              status: 'info',
              metadata: {
                agent: 'marketing_agent'
              }
            },
            {
              type: 'execution',
              timestamp: new Date(Date.now() - 7230000).toISOString(),
              description: 'Task execution failed: insufficient data',
              status: 'error',
              metadata: {
                executionTime: 1.5
              }
            }
          ],
          result: 'Error: Could not optimize marketing campaign due to insufficient historical data.'
        }
      ]
      setTasks(mockTasks)
      setLoading(false)
    }, 1000)
    
    // In a real implementation, this would be an API call:
    /*
    const fetchTasks = async () => {
      try {
        const response = await fetch('/api/tasks')
        const data = await response.json()
        setTasks(data)
      } catch (error) {
        console.error('Failed to fetch tasks:', error)
      } finally {
        setLoading(false)
      }
    }
    
    fetchTasks()
    */
  }, [])

  const handleTaskClick = (task: Task) => {
    setSelectedTask(task === selectedTask ? null : task)
  }

  const getBadgeType = (status: string) => {
    switch (status) {
      case 'completed': return 'success'
      case 'in_progress': return 'warning'
      case 'failed': return 'danger'
      default: return 'info'
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString()
  }

  return (
    <div className="tasks-page">
      <h1>{t('tasks.title')}</h1>
      
      <Card className="tasks-container">
        {loading ? (
          <div className="loading-container">
            <Spinner size="lg" />
            <p>{t('tasks.loading')}</p>
          </div>
        ) : (
          <Table>
            <thead>
              <tr>
                <th>{t('tasks.description')}</th>
                <th>{t('tasks.status')}</th>
                <th>{t('tasks.timestamp')}</th>
                <th>{t('tasks.agent')}</th>
                <th>{t('tasks.actions')}</th>
              </tr>
            </thead>
            <tbody>
              {tasks.map(task => (
                <React.Fragment key={task.id}>
                  <tr>
                    <td>{task.description}</td>
                    <td>
                      <Badge type={getBadgeType(task.status)}>
                        {t(`tasks.status.${task.status}`)}
                      </Badge>
                    </td>
                    <td>{formatDate(task.timestamp)}</td>
                    <td>{task.agent}</td>
                    <td>
                      <Button 
                        variant="secondary" 
                        size="sm"
                        onClick={() => handleTaskClick(task)}
                      >
                        {selectedTask?.id === task.id ? t('tasks.hideDetails') : t('tasks.showDetails')}
                      </Button>
                    </td>
                  </tr>
                  {selectedTask?.id === task.id && (
                    <tr>
                      <td colSpan={5}>
                        <Card className="task-details">
                          <h3>{t('tasks.details')}</h3>
                          
                          <Timeline>
                            {task.events.map((event, index) => (
                              <Timeline.Item 
                                key={index}
                                title={t(`tasks.events.${event.type}`)}
                                time={formatDate(event.timestamp)}
                                status={event.status}
                              >
                                <p>{event.description}</p>
                                {event.metadata && (
                                  <div className="event-metadata">
                                    {event.metadata.agent && (
                                      <p>{t('tasks.selectedAgent')}: {event.metadata.agent}</p>
                                    )}
                                    {event.metadata.executionTime && (
                                      <p>{t('tasks.executionTime')}: {event.metadata.executionTime}s</p>
                                    )}
                                  </div>
                                )}
                              </Timeline.Item>
                            ))}
                          </Timeline>
                          
                          {task.result && (
                            <div className="task-result">
                              <h4>{t('tasks.result')}</h4>
                              <div className="result-content">
                                {task.result}
                              </div>
                            </div>
                          )}
                        </Card>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))}
            </tbody>
          </Table>
        )}
      </Card>
    </div>
  )
}

export default TasksPage