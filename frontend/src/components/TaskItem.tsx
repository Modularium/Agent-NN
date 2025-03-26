import React from 'react';
import { Card, Badge, Button, Timeline } from '@smolitux/core';
import { Task } from '../types';
import { useTranslation } from '../utils/i18n';

interface TaskItemProps {
  task: Task;
  expanded: boolean;
  onToggle: () => void;
}

const TaskItem: React.FC<TaskItemProps> = ({ task, expanded, onToggle }) => {
  const t = useTranslation();

  const getBadgeType = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'in_progress': return 'warning';
      case 'failed': return 'danger';
      default: return 'info';
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  return (
    <div className="task-item">
      <Card className="task-card">
        <div className="task-header">
          <div className="task-title">
            <h3>{task.description}</h3>
            <Badge type={getBadgeType(task.status)}>
              {t(`tasks.status.${task.status}`)}
            </Badge>
          </div>
          
          <div className="task-meta">
            <div className="task-info">
              <span className="task-timestamp">{formatDate(task.timestamp)}</span>
              <span className="task-agent">{task.agent}</span>
            </div>
            
            <Button 
              variant="secondary" 
              size="sm"
              onClick={onToggle}
            >
              {expanded ? t('tasks.hideDetails') : t('tasks.showDetails')}
            </Button>
          </div>
        </div>
        
        {expanded && (
          <div className="task-details">
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
          </div>
        )}
      </Card>
    </div>
  );
};

export default TaskItem;