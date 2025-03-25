import React from 'react';
import { Card, Badge, Button } from '@smolitux/core';
import { Agent } from '../types';

interface AgentCardProps {
  agent: Agent;
  onClick: (agent: Agent) => void;
}

const AgentCard: React.FC<AgentCardProps> = ({ agent, onClick }) => {
  const getBadgeType = (rate: number) => {
    if (rate > 0.8) return 'success';
    if (rate > 0.5) return 'warning';
    return 'danger';
  };

  return (
    <Card className="agent-card">
      <div className="agent-card-header">
        <h3>{agent.name}</h3>
        <Badge type={getBadgeType(agent.successRate)}>
          {(agent.successRate * 100).toFixed(1)}%
        </Badge>
      </div>
      
      <div className="agent-card-content">
        <p className="agent-domain">Domain: {agent.domain}</p>
        <p className="agent-description">{agent.description}</p>
        
        <div className="agent-stats">
          <div className="stat">
            <span className="stat-label">Tasks:</span>
            <span className="stat-value">{agent.totalTasks}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Avg. Time:</span>
            <span className="stat-value">{agent.avgExecutionTime.toFixed(2)}s</span>
          </div>
          <div className="stat">
            <span className="stat-label">Documents:</span>
            <span className="stat-value">{agent.knowledgeBase.documentsCount}</span>
          </div>
        </div>
      </div>
      
      <div className="agent-card-footer">
        <Button 
          variant="secondary" 
          size="sm"
          onClick={() => onClick(agent)}
        >
          Details
        </Button>
      </div>
    </Card>
  );
};

export default AgentCard;