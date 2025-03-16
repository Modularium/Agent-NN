// monitoring/dashboard/components/panels/MonitoringPanel.tsx
import React, { useState } from 'react';
import { Cpu, Server, HardDrive, Network, RefreshCw, Download, Calendar, Clock, Activity, BarChart2, TrendingUp } from 'lucide-react';
import { useDashboard } from '../../context/DashboardContext';
import Card from '../common/Card';
import MetricCard from '../common/MetricCard';

const MonitoringPanel: React.FC = () => {
  const { systemData } = useDashboard();
  const [timeRange, setTimeRange] = useState<string>('24h');
  
  // Mock time series data
  const generateMockTimeSeriesData = (min: number, max: number, dataPoints: number, trend: 'up' | 'down' | 'stable' = 'stable', volatility: number = 0.1) => {
    const data = [];
    let value = min + Math.random() * (max - min);
    const now = new Date();
    
    for (let i = dataPoints; i > 0; i--) {
      // Generate timestamp
      const timestamp = new Date(now);
      timestamp.setMinutes(now.getMinutes() - i * 15); // 15-minute intervals
      
      // Adjust value based on trend
      if (trend === 'up') {
        value += (Math.random() * volatility * (max - min)) / dataPoints;
      } else if (trend === 'down') {
        value -= (Math.random() * volatility * (max - min)) / dataPoints;
      } else {
        value += (Math.random() * 2 - 1) * volatility * (max - min) / dataPoints;
      }
      
      // Ensure value stays within bounds
      value = Math.max(min, Math.min(max, value));
      
      data.push({
        timestamp: timestamp.toISOString(),
        value: parseFloat(value.toFixed(2))
      });
    }
    
    return data;
  };
  
  const cpuData = generateMockTimeSeriesData(10, 80, 96, 'up', 0.2); // 24 hours of 15-minute data
  const memoryData = generateMockTimeSeriesData(60, 90, 96, 'stable', 0.05);
  const gpuData = generateMockTimeSeriesData(20, 70, 96, 'up', 0.1);
  const taskData = generateMockTimeSeriesData(5, 20, 96, 'down', 0.3);
  const responseTimeData = generateMockTimeSeriesData(0.8, 2.5, 96, 'down', 0.2);
  
  // Mock agent performance data
  const agentPerformanceData = [
    { agent: 'Finance Agent', successRate: 92.5, responseTime: 1.2, tasks: 1245 },
    { agent: 'Tech Agent', successRate: 94.1, responseTime: 0.9, tasks: 2153 },
    { agent: 'Marketing Agent', successRate: 88.3, responseTime: 1.5, tasks: 987 },
    { agent: 'Web Agent', successRate: 91.2, responseTime: 1.1, tasks: 1856 },
    { agent: 'Research Agent', successRate: 89.7, responseTime: 2.3, tasks: 654 }
  ];
  
  // Filter data based on time range
  const filterDataByTimeRange = (data: any[]) => {
    const now = new Date();
    let cutoff = new Date(now);
    
    switch (timeRange) {
      case '1h':
        cutoff.setHours(now.getHours() - 1);
        break;
      case '6h':
        cutoff.setHours(now.getHours() - 6);
        break;
      case '24h':
        cutoff.setHours(now.getHours() - 24);
        break;
      case '7d':
        cutoff.setDate(now.getDate() - 7);
        break;
      case '30d':
        cutoff.setDate(now.getDate() - 30);
        break;
      default:
        cutoff.setHours(now.getHours() - 24);
    }
    
    return data.filter(item => new Date(item.timestamp) > cutoff);
  };
  
  // Simplify time series data for display
  const simplifyTimeSeriesData = (data: any[], maxPoints: number = 20) => {
    if (data.length <= maxPoints) return data;
    
    const step = Math.ceil(data.length / maxPoints);
    const result = [];
    
    for (let i = 0; i < data.length; i += step) {
      result.push(data[i]);
    }
    
    // Ensure the last point is included
    if (result[result.length - 1] !== data[data.length - 1]) {
      result.push(data[data.length - 1]);
    }
    
    return result;
  };
  
  // Format time labels
  const formatTimeLabel = (dateString: string) => {
    const date = new Date(dateString);
    if (timeRange === '1h' || timeRange === '6h') {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (timeRange === '24h') {
      return date.toLocaleTimeString([], { hour: '2-digit' });
    } else {
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
  };
  
  const filteredCpuData = filterDataByTimeRange(cpuData);
  const filteredMemoryData = filterDataByTimeRange(memoryData);
  const filteredGpuData = filterDataByTimeRange(gpuData);
  const filteredTaskData = filterDataByTimeRange(taskData);
  const filteredResponseTimeData = filterDataByTimeRange(responseTimeData);
  
  const simplifiedCpuData = simplifyTimeSeriesData(filteredCpuData);
  const simplifiedMemoryData = simplifyTimeSeriesData(filteredMemoryData);
  const simplifiedGpuData = simplifyTimeSeriesData(filteredGpuData);
  const simplifiedTaskData = simplifyTimeSeriesData(filteredTaskData);
  const simplifiedResponseTimeData = simplifyTimeSeriesData(filteredResponseTimeData);
  
  // Calculate average response time
  const avgResponseTime = responseTimeData.reduce((sum, item) => sum + item.value, 0) / responseTimeData.length;
  
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">System Monitoring</h2>
        <div className="flex space-x-3">
          <select 
            className="form-select rounded-md border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white"
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
          >
            <option value="1h">Last 1 hour</option>
            <option value="6h">Last 6 hours</option>
            <option value="24h">Last 24 hours</option>
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
          </select>
          <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 flex items-center transition">
            <Download size={18} className="mr-2" />
            Export Data
          </button>
        </div>
      </div>

      {/* System Health Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricCard
          title="CPU Usage"
          value={`${systemData?.metrics.cpu_usage || 0}`}
          unit="%"
          status={(systemData?.metrics.cpu_usage || 0) > 80 ? 'critical' : (systemData?.metrics.cpu_usage || 0) > 60 ? 'warning' : 'normal'}
          icon={<Cpu className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />}
          trend={{
            value: 5.2,
            isUpward: true,
            isGood: false
          }}
        />
        <MetricCard
          title="Memory Usage"
          value={`${systemData?.metrics.memory_usage || 0}`}
          unit="%"
          status={(systemData?.metrics.memory_usage || 0) > 80 ? 'critical' : (systemData?.metrics.memory_usage || 0) > 60 ? 'warning' : 'normal'}
          icon={<Server className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />}
          trend={{
            value: 2.1,
            isUpward: true,
            isGood: false
          }}
        />
        <MetricCard
          title="GPU Usage"
          value={`${systemData?.metrics.gpu_usage || 0}`}
          unit="%"
          status={(systemData?.metrics.gpu_usage || 0) > 80 ? 'critical' : (systemData?.metrics.gpu_usage || 0) > 60 ? 'warning' : 'normal'}
          icon={<HardDrive className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />}
          trend={{
            value: 8.5,
            isUpward: true,
            isGood: false
          }}
        />
        <MetricCard
          title="Avg Response Time"
          value={avgResponseTime.toFixed(2)}
          unit="s"
          status={avgResponseTime > 2.0 ? 'critical' : avgResponseTime > 1.5 ? 'warning' : 'normal'}
          icon={<Network className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />}
          trend={{
            value: 3.1,
            isUpward: false,
            isGood: true
          }}
        />
      </div>

      {/* Time Series Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card title="CPU & Memory Usage">
          <div className="h-64 w-full">
            {/* SVG Line Chart for CPU & Memory */}
            <svg viewBox="0 0 800 250" preserveAspectRatio="none" className="w-full h-full">
              {/* X-axis */}
              <line x1="50" y1="220" x2="750" y2="220" stroke="#E5E7EB" strokeWidth="1" className="dark:stroke-gray-700" />
              
              {/* Y-axis */}
              <line x1="50" y1="20" x2="50" y2="220" stroke="#E5E7EB" strokeWidth="1" className="dark:stroke-gray-700" />
              
              {/* Y-axis labels */}
              <text x="30" y="30" textAnchor="end" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">100%</text>
              <text x="30" y="80" textAnchor="end" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">75%</text>
              <text x="30" y="130" textAnchor="end" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">50%</text>
              <text x="30" y="180" textAnchor="end" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">25%</text>
              <text x="30" y="220" textAnchor="end" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">0%</text>
              
              {/* X-axis labels (simplified for demonstration) */}
              {simplifiedCpuData.filter((_, i) => i % 4 === 0).map((item, i, arr) => (
                <text 
                  key={i} 
                  x={50 + (i * (700 / (arr.length - 1)))} 
                  y="240" 
                  textAnchor="middle" 
                  fontSize="10"
                  fill="#6B7280"
                  className="dark:fill-gray-400"
                >
                  {formatTimeLabel(item.timestamp)}
                </text>
              ))}
              
              {/* GPU usage line */}
              <polyline
                points={simplifiedGpuData.map((point, i, arr) => {
                  const x = 50 + (i * (700 / (arr.length - 1)));
                  const y = 220 - (point.value * 2);
                  return `${x},${y}`;
                }).join(' ')}
                fill="none"
                stroke="#8B5CF6"
                strokeWidth="2"
                className="dark:stroke-purple-500"
              />
              
              {/* Task Queue bars */}
              {simplifiedTaskData.map((point, i, arr) => {
                const barWidth = 700 / arr.length * 0.6;
                const x = 50 + (i * (700 / (arr.length - 1))) - barWidth / 2;
                const barHeight = point.value * 4;
                const y = 220 - barHeight;
                return (
                  <rect
                    key={i}
                    x={x}
                    y={y}
                    width={barWidth}
                    height={barHeight}
                    fill="#F59E0B"
                    fillOpacity="0.6"
                    className="dark:fill-yellow-500 dark:fill-opacity-0.7"
                  />
                );
              })}
              
              {/* Legend */}
              <circle cx="650" cy="30" r="5" fill="#8B5CF6" className="dark:fill-purple-500" />
              <text x="660" y="35" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">GPU</text>
              <rect x="710" y="25" width="10" height="10" fill="#F59E0B" fillOpacity="0.6" className="dark:fill-yellow-500 dark:fill-opacity-0.7" />
              <text x="725" y="35" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">Tasks</text>
            </svg>
          </div>
        </Card>
      </div>

      {/* Response Time & Agent Performance */}
      <Card title="Response Time Trend">
        <div className="h-64 w-full">
          {/* SVG Line Chart for Response Time */}
          <svg viewBox="0 0 800 250" preserveAspectRatio="none" className="w-full h-full">
            {/* X-axis */}
            <line x1="50" y1="220" x2="750" y2="220" stroke="#E5E7EB" strokeWidth="1" className="dark:stroke-gray-700" />
            
            {/* Y-axis */}
            <line x1="50" y1="20" x2="50" y2="220" stroke="#E5E7EB" strokeWidth="1" className="dark:stroke-gray-700" />
            
            {/* Y-axis labels */}
            <text x="30" y="30" textAnchor="end" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">3.0s</text>
            <text x="30" y="80" textAnchor="end" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">2.5s</text>
            <text x="30" y="130" textAnchor="end" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">2.0s</text>
            <text x="30" y="180" textAnchor="end" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">1.0s</text>
            <text x="30" y="220" textAnchor="end" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">0.0s</text>
            
            {/* X-axis labels */}
            {simplifiedResponseTimeData.filter((_, i) => i % 4 === 0).map((item, i, arr) => (
              <text 
                key={i} 
                x={50 + (i * (700 / (arr.length - 1)))} 
                y="240" 
                textAnchor="middle" 
                fontSize="10" 
                fill="#6B7280"
                className="dark:fill-gray-400"
              >
                {formatTimeLabel(item.timestamp)}
              </text>
            ))}
            
            {/* Response time area */}
            <path
              d={`
                M 50 ${220 - simplifiedResponseTimeData[0].value * 80}
                ${simplifiedResponseTimeData.map((point, i, arr) => {
                  const x = 50 + (i * (700 / (arr.length - 1)));
                  const y = 220 - (point.value * 80);
                  return `L ${x} ${y}`;
                }).join(' ')}
                L 750 220
                L 50 220
                Z
              `}
              fill="#3B82F6"
              fillOpacity="0.2"
              className="dark:fill-blue-500 dark:fill-opacity-0.3"
            />
            
            {/* Response time line */}
            <polyline
              points={simplifiedResponseTimeData.map((point, i, arr) => {
                const x = 50 + (i * (700 / (arr.length - 1)));
                const y = 220 - (point.value * 80);
                return `${x},${y}`;
              }).join(' ')}
              fill="none"
              stroke="#3B82F6"
              strokeWidth="2"
              className="dark:stroke-blue-500"
            />
            
            {/* Data points */}
            {simplifiedResponseTimeData.filter((_, i) => i % 4 === 0).map((point, i, arr) => {
              const x = 50 + (i * 4 * (700 / (arr.length - 1)));
              const y = 220 - (point.value * 80);
              return (
                <circle
                  key={i}
                  cx={x}
                  cy={y}
                  r="4"
                  fill="#3B82F6"
                  className="dark:fill-blue-500"
                />
              );
            })}
            
            {/* Average line */}
            <line 
              x1="50" 
              y1={220 - (avgResponseTime * 80)} 
              x2="750" 
              y2={220 - (avgResponseTime * 80)} 
              stroke="#EF4444" 
              strokeWidth="2" 
              strokeDasharray="5,5"
              className="dark:stroke-red-500" 
            />
            <text 
              x="740" 
              y={215 - (avgResponseTime * 80)} 
              textAnchor="end" 
              fontSize="12" 
              fill="#EF4444"
              className="dark:fill-red-500"
            >
              Avg: {avgResponseTime.toFixed(2)}s
            </text>
          </svg>
        </div>
      </Card>

      {/* Agent Performance Table */}
      <Card title="Agent Performance">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead>
              <tr>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Agent</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Success Rate</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Response Time</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Tasks Completed</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Trend</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {agentPerformanceData.map((agent, index) => (
                <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                  <td className="px-3 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">{agent.agent}</td>
                  <td className="px-3 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <span className="text-sm text-gray-900 dark:text-white mr-2">{agent.successRate}%</span>
                      <div className="w-24 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div 
                          className={`h-full rounded-full ${
                            agent.successRate > 90 ? 'bg-green-500' : 
                            agent.successRate > 80 ? 'bg-yellow-500' : 'bg-red-500'
                          }`} 
                          style={{ width: `${agent.successRate}%` }}
                        ></div>
                      </div>
                    </div>
                  </td>
                  <td className="px-3 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <span className="text-sm text-gray-900 dark:text-white mr-2">{agent.responseTime}s</span>
                      <div className="w-24 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div 
                          className={`h-full rounded-full ${
                            agent.responseTime < 1.0 ? 'bg-green-500' : 
                            agent.responseTime < 2.0 ? 'bg-yellow-500' : 'bg-red-500'
                          }`} 
                          style={{ width: `${Math.min(100, agent.responseTime * 50)}%` }}
                        ></div>
                      </div>
                    </div>
                  </td>
                  <td className="px-3 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <span className="text-sm text-gray-900 dark:text-white mr-2">{agent.tasks.toLocaleString()}</span>
                      <div className="w-24 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-indigo-500 h-full rounded-full" 
                          style={{ width: `${(agent.tasks / agentPerformanceData.reduce((max, a) => Math.max(max, a.tasks), 0)) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                    {index % 2 === 0 ? (
                      <TrendingUp size={20} className="text-green-500" />
                    ) : (
                      <Activity size={20} className="text-blue-500" />
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Historical Monitoring */}
      <Card>
        <div className="flex justify-between mb-4">
          <h3 className="font-bold text-gray-900 dark:text-white">Historical Performance</h3>
          <div className="flex items-center space-x-4">
            <div className="flex items-center">
              <Calendar size={16} className="text-gray-500 dark:text-gray-400 mr-2" />
              <span className="text-sm text-gray-500 dark:text-gray-400">
                {new Date().toLocaleDateString([], {year: 'numeric', month: 'short', day: 'numeric'})}
              </span>
            </div>
            <div className="flex items-center">
              <Clock size={16} className="text-gray-500 dark:text-gray-400 mr-2" />
              <span className="text-sm text-gray-500 dark:text-gray-400">
                {new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
              </span>
            </div>
            <button className="p-2 rounded-md border border-gray-300 dark:border-gray-700 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300">
              <RefreshCw size={16} />
            </button>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="border rounded-lg p-4 dark:border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <h4 className="font-medium text-gray-900 dark:text-white">Uptime</h4>
              <span className="text-green-500 font-medium">99.97%</span>
            </div>
            <div className="h-24 flex items-end space-x-1">
              {Array.from({ length: 28 }).map((_, i) => {
                const height = 90 + Math.random() * 10;
                return (
                  <div 
                    key={i} 
                    className="bg-green-500 dark:bg-green-600 rounded-t" 
                    style={{ 
                      height: `${height}%`, 
                      width: `${100 / 28}%` 
                    }}
                  ></div>
                );
              })}
            </div>
            <div className="flex justify-between mt-2 text-xs text-gray-500 dark:text-gray-400">
              <span>Feb 15</span>
              <span>Mar 15</span>
            </div>
          </div>
          
          <div className="border rounded-lg p-4 dark:border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <h4 className="font-medium text-gray-900 dark:text-white">Error Rate</h4>
              <span className="text-yellow-500 font-medium">2.1%</span>
            </div>
            <div className="h-24 flex items-end space-x-1">
              {Array.from({ length: 28 }).map((_, i) => {
                const height = Math.random() * 20;
                return (
                  <div 
                    key={i} 
                    className="bg-yellow-500 dark:bg-yellow-600 rounded-t" 
                    style={{ 
                      height: `${height}%`, 
                      width: `${100 / 28}%` 
                    }}
                  ></div>
                );
              })}
            </div>
            <div className="flex justify-between mt-2 text-xs text-gray-500 dark:text-gray-400">
              <span>Feb 15</span>
              <span>Mar 15</span>
            </div>
          </div>
          
          <div className="border rounded-lg p-4 dark:border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <h4 className="font-medium text-gray-900 dark:text-white">Task Throughput</h4>
              <span className="text-blue-500 font-medium">152/hour</span>
            </div>
            <div className="h-24 flex items-end space-x-1">
              {Array.from({ length: 28 }).map((_, i) => {
                const height = 30 + Math.random() * 70;
                return (
                  <div 
                    key={i} 
                    className="bg-blue-500 dark:bg-blue-600 rounded-t" 
                    style={{ 
                      height: `${height}%`, 
                      width: `${100 / 28}%` 
                    }}
                  ></div>
                );
              })}
            </div>
            <div className="flex justify-between mt-2 text-xs text-gray-500 dark:text-gray-400">
              <span>Feb 15</span>
              <span>Mar 15</span>
            </div>
          </div>
        </div>
        
        <div className="flex">
          <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 transition">
            View Detailed Reports
          </button>
        </div>
      </Card>
    </div>
  );
};

export default MonitoringPanel; 
                  fill="#6B7280"
                  className="dark:fill-gray-400"
                >
                  {formatTimeLabel(item.timestamp)}
                </text>
              ))}
              
              {/* CPU usage line */}
              <polyline
                points={simplifiedCpuData.map((point, i, arr) => {
                  const x = 50 + (i * (700 / (arr.length - 1)));
                  const y = 220 - (point.value * 2);
                  return `${x},${y}`;
                }).join(' ')}
                fill="none"
                stroke="#4F46E5"
                strokeWidth="2"
                className="dark:stroke-indigo-500"
              />
              
              {/* Memory usage line */}
              <polyline
                points={simplifiedMemoryData.map((point, i, arr) => {
                  const x = 50 + (i * (700 / (arr.length - 1)));
                  const y = 220 - (point.value * 2);
                  return `${x},${y}`;
                }).join(' ')}
                fill="none"
                stroke="#10B981"
                strokeWidth="2"
                className="dark:stroke-green-500"
              />
              
              {/* Legend */}
              <circle cx="680" cy="30" r="5" fill="#4F46E5" className="dark:fill-indigo-500" />
              <text x="690" y="35" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">CPU</text>
              <circle cx="730" cy="30" r="5" fill="#10B981" className="dark:fill-green-500" />
              <text x="740" y="35" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">Memory</text>
            </svg>
          </div>
        </Card>
        
        <Card title="GPU Usage & Task Queue">
          <div className="h-64 w-full">
            {/* SVG Line Chart for GPU & Task Queue */}
            <svg viewBox="0 0 800 250" preserveAspectRatio="none" className="w-full h-full">
              {/* X-axis */}
              <line x1="50" y1="220" x2="750" y2="220" stroke="#E5E7EB" strokeWidth="1" className="dark:stroke-gray-700" />
              
              {/* Y-axis for GPU */}
              <line x1="50" y1="20" x2="50" y2="220" stroke="#E5E7EB" strokeWidth="1" className="dark:stroke-gray-700" />
              
              {/* Y-axis labels for GPU */}
              <text x="30" y="30" textAnchor="end" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">100%</text>
              <text x="30" y="80" textAnchor="end" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">75%</text>
              <text x="30" y="130" textAnchor="end" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">50%</text>
              <text x="30" y="180" textAnchor="end" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">25%</text>
              <text x="30" y="220" textAnchor="end" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">0%</text>
              
              {/* Y-axis for Task Queue */}
              <line x1="750" y1="20" x2="750" y2="220" stroke="#E5E7EB" strokeWidth="1" className="dark:stroke-gray-700" />
              
              {/* Y-axis labels for Task Queue */}
              <text x="770" y="30" textAnchor="start" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">50</text>
              <text x="770" y="80" textAnchor="start" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">40</text>
              <text x="770" y="130" textAnchor="start" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">30</text>
              <text x="770" y="180" textAnchor="start" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">10</text>
              <text x="770" y="220" textAnchor="start" fontSize="12" fill="#6B7280" className="dark:fill-gray-400">0</text>
              
              {/* X-axis labels (simplified for demonstration) */}
              {simplifiedGpuData.filter((_, i) => i % 4 === 0).map((item, i, arr) => (
                <text 
                  key={i} 
                  x={50 + (i * (700 / (arr.length - 1)))} 
                  y="240" 
                  textAnchor="middle" 
                  fontSize="10"
