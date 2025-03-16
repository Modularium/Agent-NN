// monitoring/dashboard/components/panels/SystemOverviewPanel.tsx
import React from 'react';
import { Cpu, Server, HardDrive, Network } from 'lucide-react';
import { useDashboard } from '../../context/DashboardContext';
import Card from '../common/Card';
import MetricCard from '../common/MetricCard';
import StatusBadge from '../common/StatusBadge';
import { formatRelativeTime } from '../../utils/formatters';

const SystemOverviewPanel: React.FC = () => {
  const { systemData } = useDashboard();
  
  if (!systemData) {
    return <div>No system data available</div>;
  }
  
  const determineStatus = (value: number): 'normal' | 'warning' | 'critical' => {
    if (value > 90) return 'critical';
    if (value > 70) return 'warning';
    return 'normal';
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">System Overview</h2>
        <div className="flex space-x-2">
          <StatusBadge status="System Online" />
        </div>
      </div>

      {/* Resource Usage Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="CPU Usage"
          value={`${systemData.metrics.cpu_usage}`}
          unit="%"
          status={determineStatus(systemData.metrics.cpu_usage)}
          icon={<Cpu className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />}
        />
        <MetricCard
          title="Memory Usage"
          value={`${systemData.metrics.memory_usage}`}
          unit="%"
          status={determineStatus(systemData.metrics.memory_usage)}
          icon={<Server className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />}
        />
        <MetricCard
          title="GPU Usage"
          value={`${systemData.metrics.gpu_usage}`}
          unit="%"
          status={determineStatus(systemData.metrics.gpu_usage)}
          icon={<Cpu className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />}
        />
        <MetricCard
          title="Disk Usage"
          value={`${systemData.metrics.disk_usage}`}
          unit="%"
          status={determineStatus(systemData.metrics.disk_usage)}
          icon={<HardDrive className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />}
        />
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricCard
          title="Active Agents"
          value={systemData.metrics.active_agents}
          className="md:col-span-1"
        />
        <MetricCard
          title="Tasks in Queue"
          value={systemData.metrics.task_queue_size}
          className="md:col-span-1"
        />
        <MetricCard
          title="Avg Response Time"
          value={systemData.metrics.avg_response_time.toFixed(2)}
          unit="s"
          className="md:col-span-1"
        />
      </div>

      {/* Active Tasks & Components */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Active Tasks">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead>
                <tr>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">ID</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Type</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Agent</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Duration</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {systemData.activeTasks.map((task) => (
                  <tr key={task.id} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                    <td className="px-3 py-2 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">{task.id}</td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{task.type}</td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{task.agent}</td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm">
                      <StatusBadge status={task.status} />
                    </td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{task.duration}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="System Components">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead>
                <tr>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Component</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Version</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Last Updated</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {systemData.components.map((component, index) => (
                  <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                    <td className="px-3 py-2 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">{component.name}</td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm">
                      <StatusBadge status={component.status} />
                    </td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{component.version}</td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{component.lastUpdated}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      {/* System Configuration */}
      <Card title="System Configuration">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Maximum Concurrent Tasks</label>
              <div className="flex items-center">
                <input type="number" className="form-input rounded-md w-24 border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white" defaultValue="10" />
                <button className="ml-2 px-3 py-1 bg-indigo-100 text-indigo-700 dark:bg-indigo-900 dark:text-indigo-300 rounded text-sm">
                  Update
                </button>
              </div>
            </div>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Task Timeout (seconds)</label>
              <div className="flex items-center">
                <input type="number" className="form-input rounded-md w-24 border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white" defaultValue="300" />
                <button className="ml-2 px-3 py-1 bg-indigo-100 text-indigo-700 dark:bg-indigo-900 dark:text-indigo-300 rounded text-sm">
                  Update
                </button>
              </div>
            </div>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Logging Level</label>
              <div className="flex items-center">
                <select className="form-select rounded-md w-36 border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white">
                  <option>INFO</option>
                  <option>DEBUG</option>
                  <option>WARNING</option>
                  <option>ERROR</option>
                </select>
                <button className="ml-2 px-3 py-1 bg-indigo-100 text-indigo-700 dark:bg-indigo-900 dark:text-indigo-300 rounded text-sm">
                  Update
                </button>
              </div>
            </div>
          </div>
          <div>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Cache Size (MB)</label>
              <div className="flex items-center">
                <input type="number" className="form-input rounded-md w-24 border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white" defaultValue="1024" />
                <button className="ml-2 px-3 py-1 bg-indigo-100 text-indigo-700 dark:bg-indigo-900 dark:text-indigo-300 rounded text-sm">
                  Update
                </button>
              </div>
            </div>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Monitoring Interval (seconds)</label>
              <div className="flex items-center">
                <input type="number" className="form-input rounded-md w-24 border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white" defaultValue="60" />
                <button className="ml-2 px-3 py-1 bg-indigo-100 text-indigo-700 dark:bg-indigo-900 dark:text-indigo-300 rounded text-sm">
                  Update
                </button>
              </div>
            </div>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Backup Interval (hours)</label>
              <div className="flex items-center">
                <input type="number" className="form-input rounded-md w-24 border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white" defaultValue="24" />
                <button className="ml-2 px-3 py-1 bg-indigo-100 text-indigo-700 dark:bg-indigo-900 dark:text-indigo-300 rounded text-sm">
                  Update
                </button>
              </div>
            </div>
          </div>
        </div>

        <div className="flex space-x-3 mt-4">
          <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600">
            Save All Changes
          </button>
          <button className="px-4 py-2 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600">
            Reset to Defaults
          </button>
          <button className="px-4 py-2 bg-red-100 text-red-700 rounded-md hover:bg-red-200 dark:bg-red-900/30 dark:text-red-400 dark:hover:bg-red-900/50">
            Clear Cache
          </button>
        </div>
      </Card>
    </div>
  );
};

export default SystemOverviewPanel;
