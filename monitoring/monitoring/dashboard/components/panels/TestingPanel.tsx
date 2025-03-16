// monitoring/dashboard/components/panels/TestingPanel.tsx
import React, { useState } from 'react';
import { GitBranch, Plus, TrendingUp, Clock, Award, CheckCircle, AlertTriangle } from 'lucide-react';
import { useDashboard } from '../../context/DashboardContext';
import Card from '../common/Card';
import StatusBadge from '../common/StatusBadge';

const TestingPanel: React.FC = () => {
  const { testResults } = useDashboard();
  const [selectedTest, setSelectedTest] = useState<string | null>(null);
  
  // Mock test details data
  const testDetails = {
    id: 'test-001',
    name: 'Prompt Optimization',
    status: 'completed',
    startDate: '2025-03-14T08:00:00Z',
    endDate: '2025-03-15T08:00:00Z',
    variants: [
      {
        name: 'Variant A (Baseline)',
        description: 'Analyze the following financial data and provide insights on investment opportunities.',
        metrics: {
          successRate: 78.5,
          responseTime: 1.8,
          userSatisfaction: 3.9
        }
      },
      {
        name: 'Variant B (Winner)',
        description: 'Given the financial data below, identify potential investment opportunities, considering risk tolerance and expected returns.',
        metrics: {
          successRate: 91.0,
          responseTime: 1.5,
          userSatisfaction: 4.5
        }
      }
    ]
  };
  
  // Function to format timestamp
  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };
  
  // Function to calculate improvement percentage
  const calculateImprovement = (baselineValue: number, newValue: number): string => {
    const improvement = ((newValue - baselineValue) / baselineValue) * 100;
    return improvement > 0 ? `+${improvement.toFixed(1)}%` : `${improvement.toFixed(1)}%`;
  };
  
  // Handle test selection
  const handleTestClick = (testId: string) => {
    setSelectedTest(testId === selectedTest ? null : testId);
  };
  
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">A/B Testing</h2>
        <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 flex items-center transition">
          <Plus size={18} className="mr-2" />
          Create New Test
        </button>
      </div>

      {/* Test Results */}
      <Card title="Test Results">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead>
              <tr>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Name</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Variants</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Winner</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Improvement</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {testResults.map((test, index) => (
                <tr 
                  key={index} 
                  className={`hover:bg-gray-50 dark:hover:bg-gray-800 ${selectedTest === test.id ? 'bg-indigo-50 dark:bg-indigo-900/20' : ''}`}
                  onClick={() => handleTestClick(test.id)}
                >
                  <td className="px-3 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">{test.name}</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">
                    <StatusBadge status={test.status} />
                  </td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{test.variants}</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{test.winner}</td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {test.improvement !== '-' && (
                      <span className={`flex items-center ${
                        test.improvement.startsWith('+') ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                      }`}>
                        {test.improvement.startsWith('+') ? <TrendingUp size={16} className="mr-1" /> : null}
                        {test.improvement}
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-4 whitespace-nowrap text-sm">
                    <div className="flex space-x-2">
                      <button 
                        className="text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-300"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleTestClick(test.id);
                        }}
                      >
                        View
                      </button>
                      {test.status !== 'completed' && (
                        <button 
                          className="text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300"
                          onClick={(e) => e.stopPropagation()}
                        >
                          Stop
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Test Details */}
      {selectedTest && (
        <Card title={`Test Details: ${testDetails.name}`}>
          <div className="flex justify-between items-center mb-4">
            <div>
              <div className="flex items-center">
                <StatusBadge status={testDetails.status} className="mr-2" />
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {formatTimestamp(testDetails.startDate)} - {formatTimestamp(testDetails.endDate)}
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <button className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded text-sm dark:bg-indigo-900/30 dark:text-indigo-400">
                Export Results
              </button>
              {testDetails.status === 'completed' && (
                <button className="px-3 py-1 bg-green-100 text-green-700 rounded text-sm dark:bg-green-900/30 dark:text-green-400 flex items-center">
                  <Award size={16} className="mr-1" />
                  Apply Winner
                </button>
              )}
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {testDetails.variants.map((variant, index) => (
              <div 
                key={index} 
                className={`border rounded-lg p-4 ${
                  variant.name.includes('Winner') 
                    ? 'bg-green-50 border-green-200 dark:bg-green-900/20 dark:border-green-900' 
                    : 'dark:border-gray-700'
                }`}
              >
                <div className="flex justify-between items-start mb-3">
                  <h3 className={`font-semibold ${
                    variant.name.includes('Winner')
                      ? 'text-green-800 dark:text-green-400'
                      : 'text-gray-900 dark:text-white'
                  }`}>{variant.name}</h3>
                  {variant.name.includes('Winner') && (
                    <Award size={18} className="text-green-600 dark:text-green-400" />
                  )}
                </div>
                
                <div className="border rounded-lg p-3 mb-4 bg-gray-50 dark:bg-gray-800 dark:border-gray-700">
                  <p className="text-sm font-mono text-gray-700 dark:text-gray-300">{variant.description}</p>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600 dark:text-gray-400">Success Rate</span>
                      <span className={`${
                        variant.name.includes('Winner') ? 'text-green-600 dark:text-green-400 font-medium' : 'text-gray-900 dark:text-white'
                      }`}>{variant.metrics.successRate}%</span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                      <div 
                        className={`h-full rounded-full ${
                          variant.name.includes('Winner') ? 'bg-green-500' : 'bg-blue-500'
                        }`} 
                        style={{ width: `${variant.metrics.successRate}%` }}
                      ></div>
                    </div>
                    {index === 1 && (
                      <div className="text-xs text-green-600 dark:text-green-400 text-right mt-1">
                        {calculateImprovement(testDetails.variants[0].metrics.successRate, variant.metrics.successRate)} improvement
                      </div>
                    )}
                  </div>
                  
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600 dark:text-gray-400">Response Time</span>
                      <span className={`${
                        variant.name.includes('Winner') ? 'text-green-600 dark:text-green-400 font-medium' : 'text-gray-900 dark:text-white'
                      }`}>{variant.metrics.responseTime}s</span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                      <div 
                        className={`h-full rounded-full ${
                          variant.name.includes('Winner') ? 'bg-green-500' : 'bg-blue-500'
                        }`} 
                        style={{ width: `${(variant.metrics.responseTime / 3) * 100}%` }}
                      ></div>
                    </div>
                    {index === 1 && (
                      <div className="text-xs text-green-600 dark:text-green-400 text-right mt-1">
                        {calculateImprovement(testDetails.variants[0].metrics.responseTime, variant.metrics.responseTime)} improvement
                      </div>
                    )}
                  </div>
                  
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600 dark:text-gray-400">User Satisfaction</span>
                      <span className={`${
                        variant.name.includes('Winner') ? 'text-green-600 dark:text-green-400 font-medium' : 'text-gray-900 dark:text-white'
                      }`}>{variant.metrics.userSatisfaction}/5</span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                      <div 
                        className={`h-full rounded-full ${
                          variant.name.includes('Winner') ? 'bg-green-500' : 'bg-blue-500'
                        }`} 
                        style={{ width: `${(variant.metrics.userSatisfaction / 5) * 100}%` }}
                      ></div>
                    </div>
                    {index === 1 && (
                      <div className="text-xs text-green-600 dark:text-green-400 text-right mt-1">
                        {calculateImprovement(testDetails.variants[0].metrics.userSatisfaction, variant.metrics.userSatisfaction)} improvement
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          <div className="mt-6 border-t dark:border-gray-700 pt-4">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">Test Conclusions</h3>
            <div className="mb-3 flex items-start">
              <CheckCircle size={18} className="text-green-600 dark:text-green-500 mt-0.5 mr-2 flex-shrink-0" />
              <div>
                <p className="text-gray-700 dark:text-gray-300">
                  Variant B showed a significant improvement in all measured metrics compared to the baseline.
                </p>
              </div>
            </div>
            <div className="mb-3 flex items-start">
              <CheckCircle size={18} className="text-green-600 dark:text-green-500 mt-0.5 mr-2 flex-shrink-0" />
              <div>
                <p className="text-gray-700 dark:text-gray-300">
                  The more detailed prompt with specific guidance led to more accurate and focused responses.
                </p>
              </div>
            </div>
            <div className="flex items-start">
              <AlertTriangle size={18} className="text-yellow-600 dark:text-yellow-500 mt-0.5 mr-2 flex-shrink-0" />
              <div>
                <p className="text-gray-700 dark:text-gray-300">
                  Consider running additional tests with different domains to ensure the improvements are consistent across different use cases.
                </p>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Create New Test Form */}
      <Card title="Create New Test">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Test Name</label>
            <input 
              type="text" 
              className="form-input rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white" 
              placeholder="E.g., Prompt Optimization"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Test Type</label>
            <select className="form-select rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white">
              <option>Prompt Comparison</option>
              <option>Model Comparison</option>
              <option>Parameter Tuning</option>
              <option>Knowledge Source Comparison</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Variants</label>
            <div className="space-y-3">
              <div className="border rounded-lg p-3 dark:border-gray-700">
                <h4 className="font-medium text-gray-900 dark:text-white mb-2">Variant A (Baseline)</h4>
                <textarea 
                  className="form-textarea rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white" 
                  rows={3} 
                  placeholder="Enter baseline prompt or configuration"
                ></textarea>
              </div>
              
              <div className="border rounded-lg p-3 dark:border-gray-700">
                <h4 className="font-medium text-gray-900 dark:text-white mb-2">Variant B</h4>
                <textarea 
                  className="form-textarea rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white" 
                  rows={3} 
                  placeholder="Enter test prompt or configuration"
                ></textarea>
              </div>
              
              <button className="text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-300 text-sm flex items-center">
                <Plus size={16} className="mr-1" />
                Add Another Variant
              </button>
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Metrics to Track</label>
            <div className="space-y-2">
              <div className="flex items-center">
                <input type="checkbox" defaultChecked className="mr-2" />
                <span className="text-gray-700 dark:text-gray-300">Success Rate</span>
              </div>
              <div className="flex items-center">
                <input type="checkbox" defaultChecked className="mr-2" />
                <span className="text-gray-700 dark:text-gray-300">Response Time</span>
              </div>
              <div className="flex items-center">
                <input type="checkbox" className="mr-2" />
                <span className="text-gray-700 dark:text-gray-300">User Satisfaction</span>
              </div>
              <div className="flex items-center">
                <input type="checkbox" className="mr-2" />
                <span className="text-gray-700 dark:text-gray-300">Resource Usage</span>
              </div>
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Test Duration</label>
            <select className="form-select rounded-md w-full border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white">
              <option>1 day</option>
              <option>3 days</option>
              <option>7 days</option>
              <option>14 days</option>
              <option>30 days</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Traffic Distribution</label>
            <div className="flex items-center space-x-3">
              <div className="flex-1">
                <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Variant A: 50%</label>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div className="bg-blue-500 h-2 rounded-full" style={{ width: '50%' }}></div>
                </div>
              </div>
              <div className="flex-1">
                <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Variant B: 50%</label>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div className="bg-blue-500 h-2 rounded-full" style={{ width: '50%' }}></div>
                </div>
              </div>
              <div className="w-12 text-center">
                <button className="text-xs text-indigo-600 dark:text-indigo-400">Edit</button>
              </div>
            </div>
          </div>
          
          <div className="flex space-x-3 pt-4">
            <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 transition">
              Start Test
            </button>
            <button className="px-4 py-2 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600 transition">
              Save Draft
            </button>
          </div>
        </div>
      </Card>
      
      {/* Best Practices */}
      <Card title="A/B Testing Best Practices">
        <div className="space-y-4">
          <div className="flex items-start">
            <div className="flex-shrink-0 bg-indigo-100 dark:bg-indigo-900/30 rounded-full p-1 mr-3">
              <GitBranch size={18} className="text-indigo-600 dark:text-indigo-400" />
            </div>
            <div>
              <h3 className="font-medium text-gray-900 dark:text-white">Test One Variable at a Time</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                To accurately measure impact, change only one variable between variants.
              </p>
            </div>
          </div>
          
          <div className="flex items-start">
            <div className="flex-shrink-0 bg-indigo-100 dark:bg-indigo-900/30 rounded-full p-1 mr-3">
              <Clock size={18} className="text-indigo-600 dark:text-indigo-400" />
            </div>
            <div>
              <h3 className="font-medium text-gray-900 dark:text-white">Run Tests Long Enough</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Ensure your test duration allows for statistical significance. Usually 7-14 days is recommended.
              </p>
            </div>
          </div>
          
          <div className="flex items-start">
            <div className="flex-shrink-0 bg-indigo-100 dark:bg-indigo-900/30 rounded-full p-1 mr-3">
              <TrendingUp size={18} className="text-indigo-600 dark:text-indigo-400" />
            </div>
            <div>
              <h3 className="font-medium text-gray-900 dark:text-white">Focus on Meaningful Metrics</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Track metrics that directly relate to your goals, not just vanity metrics.
              </p>
            </div>
          </div>
          
          <div className="flex items-start">
            <div className="flex-shrink-0 bg-indigo-100 dark:bg-indigo-900/30 rounded-full p-1 mr-3">
              <Award size={18} className="text-indigo-600 dark:text-indigo-400" />
            </div>
            <div>
              <h3 className="font-medium text-gray-900 dark:text-white">Document and Share Results</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Keep a record of all tests and share findings with your team to build institutional knowledge.
              </p>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default TestingPanel;
