// monitoring/dashboard/hooks/useAgentData.ts
import { useState, useEffect, useCallback } from 'react';
import { Agent, AgentDetail } from '../types/system';
import { fetchSystemData, fetchAgentDetails, updateAgentStatus, createAgent } from '../utils/api';
import useRefreshInterval from './useRefreshInterval';

interface UseAgentDataOptions {
  refreshInterval?: number;
  autoRefresh?: boolean;
  initialFetch?: boolean;
  agentName?: string; // Optional, to fetch specific agent details
}

/**
 * Hook for managing agent data
 */
const useAgentData = (options: UseAgentDataOptions = {}) => {
  const {
    refreshInterval = 30000,
    autoRefresh = true,
    initialFetch = true,
    agentName
  } = options;

  const [agents, setAgents] = useState<Agent[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<AgentDetail | null>(null);
  const [loading, setLoading] = useState(initialFetch);
  const [error, setError] = useState<Error | null>(null);

  // Fetch all agents function
  const fetchAgents = useCallback(async () => {
    try {
      const data = await fetchSystemData();
      setAgents(data.agents);
      return data.agents;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to fetch agents data');
      setError(error);
      throw error;
    }
  }, []);

  // Fetch specific agent details
  const fetchAgentDetail = useCallback(async (name: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const agentDetail = await fetchAgentDetails(name);
      setSelectedAgent(agentDetail);
      return agentDetail;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(`Failed to fetch details for agent ${name}`);
      setError(error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, []);

  // Create a new agent
  const addAgent = useCallback(async (agentData: Partial<Agent>) => {
    setLoading(true);
    setError(null);
    
    try {
      const newAgent = await createAgent(agentData);
      // Update the agents list with the new agent
      setAgents(prevAgents => [...prevAgents, newAgent as Agent]);
      return newAgent;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to create new agent');
      setError(error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, []);

  // Update agent status
  const setAgentStatus = useCallback(async (agentName: string, status: 'active' | 'idle') => {
    setLoading(true);
    setError(null);
    
    try {
      const updatedAgent = await updateAgentStatus(agentName, status);
      
      // Update both the agents list and selected agent if it matches
      setAgents(prevAgents => 
        prevAgents.map(agent => 
          agent.name === agentName ? { ...agent, status } : agent
        )
      );
      
      if (selectedAgent && selectedAgent.name === agentName) {
        setSelectedAgent(prev => prev ? { ...prev, status } : null);
      }
      
      return updatedAgent;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(`Failed to update status for agent ${agentName}`);
      setError(error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [selectedAgent]);

  // Set up refresh interval for all agents
  const { 
    isRefreshing,
    lastRefreshed,
    refresh: refreshAgents,
    startRefresh,
    stopRefresh,
    isActive
  } = useRefreshInterval(fetchAgents, {
    intervalMs: refreshInterval,
    autostart: autoRefresh && !agentName, // Only auto-refresh all agents if not focusing on a specific one
    immediate: initialFetch && !agentName,
    onError: setError
  });

  // On initial mount, fetch data
  useEffect(() => {
    if (initialFetch) {
      setLoading(true);
      
      if (agentName) {
        // Fetch specific agent details if agentName is provided
        fetchAgentDetail(agentName)
          .catch(err => {
            console.error(`Error fetching details for agent ${agentName}:`, err);
          })
          .finally(() => {
            setLoading(false);
          });
      } else {
        // Otherwise fetch all agents
        fetchAgents()
          .catch(err => {
            console.error('Error fetching agents data:', err);
          })
          .finally(() => {
            setLoading(false);
          });
      }
    }
  }, [initialFetch, agentName, fetchAgents, fetchAgentDetail]);

  // Filtering and sorting helpers
  const getActiveAgents = useCallback(() => {
    return agents.filter(agent => agent.status === 'active');
  }, [agents]);

  const sortAgentsByMetric = useCallback((metric: keyof Agent, ascending = true) => {
    return [...agents].sort((a, b) => {
      if (ascending) {
        return a[metric] > b[metric] ? 1 : -1;
      } else {
        return a[metric] < b[metric] ? 1 : -1;
      }
    });
  }, [agents]);

  return {
    agents,
    selectedAgent,
    loading: loading || isRefreshing,
    error,
    lastRefreshed,
    
    // Actions
    fetchAgents: refreshAgents,
    fetchAgentDetail,
    addAgent,
    setAgentStatus,
    selectAgent: fetchAgentDetail,
    
    // Auto-refresh controls
    startAutoRefresh: startRefresh,
    stopAutoRefresh: stopRefresh,
    isAutoRefreshActive: isActive,
    
    // Helper functions
    getActiveAgents,
    sortAgentsByMetric,
    getAgentByName: (name: string) => agents.find(agent => agent.name === name) || null
  };
};

export default useAgentData;
