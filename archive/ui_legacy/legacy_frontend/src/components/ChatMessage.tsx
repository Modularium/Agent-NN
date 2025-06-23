import React from 'react';
import { Card, Alert } from '@smolitux/core';
import { Message } from '../types';

interface ChatMessageProps {
  message: Message;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  return (
    <div className={`message ${message.role}`}>
      {message.error ? (
        <Alert type="error">{message.content}</Alert>
      ) : (
        <Card className={`message-card message-${message.role}`}>
          <div className="message-content">{message.content}</div>
          {message.metadata && (
            <div className="message-metadata">
              {message.metadata.agent && (
                <span>Processed by: {message.metadata.agent}</span>
              )}
              {message.metadata.executionTime && (
                <span> ({message.metadata.executionTime.toFixed(2)}s)</span>
              )}
            </div>
          )}
        </Card>
      )}
    </div>
  );
};

export default ChatMessage;