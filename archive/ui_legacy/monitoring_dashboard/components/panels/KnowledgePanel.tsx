// monitoring/dashboard/components/panels/KnowledgePanel.tsx
import React, { useState } from 'react';
import { Database, UploadCloud, FileText, Search, Filter, Plus, RefreshCw, Trash2, Edit, Eye } from 'lucide-react';
import { useDashboard } from '../../context/DashboardContext';
import Card from '../common/Card';
import StatusBadge from '../common/StatusBadge';
import MetricCard from '../common/MetricCard';

const KnowledgePanel: React.FC = () => {
  const { knowledgeBases } = useDashboard();
  const [activeTab, setActiveTab] = useState<'browse' | 'upload' | 'settings'>('browse');
  const [selectedKB, setSelectedKB] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  
  // Mock document data
  const documents = [
    { id: 'doc-1', title: 'Financial Report Q4 2024', type: 'PDF', size: 2.4, uploadedAt: '2025-02-10T12:30:00Z', lastAccessed: '2025-03-15T08:45:00Z' },
    { id: 'doc-2', title: 'Market Analysis 2025', type: 'DOCX', size: 1.8, uploadedAt: '2025-02-15T14:20:00Z', lastAccessed: '2025-03-14T10:30:00Z' },
    { id: 'doc-3', title: 'Product Roadmap', type: 'PPTX', size: 4.5, uploadedAt: '2025-02-20T09:15:00Z', lastAccessed: '2025-03-16T09:15:00Z' },
    { id: 'doc-4', title: 'Competitor Analysis', type: 'PDF', size: 3.2, uploadedAt: '2025-02-25T16:45:00Z', lastAccessed: '2025-03-12T14:20:00Z' },
    { id: 'doc-5', title: 'Customer Survey Results', type: 'CSV', size: 0.8, uploadedAt: '2025-03-01T11:30:00Z', lastAccessed: '2025-03-10T11:10:00Z' },
  ];

  // Mock upload handler
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      // Simulate upload progress
      setUploadProgress(0);
      const interval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev !== null && prev >= 100) {
            clearInterval(interval);
            setTimeout(() => setUploadProgress(null), 1000);
            return 100;
          }
          return prev !== null ? prev + 10 : null;
        });
      }, 300);
    }
  };

  // Calculate total documents and storage
  const totalDocuments = knowledgeBases.reduce((total, kb) => total + kb.documents, 0);
  const totalStorage = knowledgeBases.reduce((total, kb) => {
    const size = parseFloat(kb.size.replace(' GB', ''));
    return total + size;
  }, 0).toFixed(1);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">Knowledge Base Management</h2>
        <div className="flex space-x-3">
          <button
            className="px-4 py-2 bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200 dark:bg-blue-900/30 dark:text-blue-300 dark:hover:bg-blue-900/50 flex items-center transition"
            onClick={() => setActiveTab('upload')}
          >
            <UploadCloud size={18} className="mr-2" />
            Import Documents
          </button>
          <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 flex items-center transition">
            <Database size={18} className="mr-2" />
            New Knowledge Base
          </button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-gray-200 dark:border-gray-700">
        <button
          className={`px-4 py-2 font-medium ${
            activeTab === 'browse'
              ? 'border-b-2 border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('browse')}
        >
          Browse Knowledge Bases
        </button>
        <button
          className={`px-4 py-2 font-medium ${
            activeTab === 'upload'
              ? 'border-b-2 border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('upload')}
        >
          Upload Documents
        </button>
        <button
          className={`px-4 py-2 font-medium ${
            activeTab === 'settings'
              ? 'border-b-2 border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('settings')}
        >
          Settings
        </button>
      </div>

      {/* Knowledge Base Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricCard
          title="Total Documents"
          value={totalDocuments.toLocaleString()}
          icon={<FileText className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />}
        />
        <MetricCard
          title="Total Storage"
          value={totalStorage}
          unit="GB"
          icon={<Database className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />}
        />
        <MetricCard
          title="Knowledge Bases"
          value={knowledgeBases.length}
          icon={<Database className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />}
        />
      </div>

      {/* Browse Knowledge Bases Tab */}
      {activeTab === 'browse' && (
        <>
          <Card>
            <div className="flex justify-between items-center mb-4">
              <h3 className="font-bold text-gray-900 dark:text-white">Knowledge Bases</h3>
              <div className="flex items-center space-x-2">
                <div className="relative">
                  <input
                    type="text"
                    className="pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white rounded-md"
                    placeholder="Search knowledge bases..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Search size={18} className="text-gray-400" />
                  </div>
                </div>
                <button className="p-2 border border-gray-300 dark:border-gray-700 rounded-md text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300">
                  <Filter size={18} />
                </button>
                <button className="p-2 border border-gray-300 dark:border-gray-700 rounded-md text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300">
                  <RefreshCw size={18} />
                </button>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                <thead>
                  <tr>
                    <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Name</th>
                    <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Documents</th>
                    <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Size</th>
                    <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Last Updated</th>
                    <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</th>
                    <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {knowledgeBases.map((kb, index) => (
                    <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                      <td className="px-3 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">{kb.name}</td>
                      <td className="px-3 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{kb.documents.toLocaleString()}</td>
                      <td className="px-3 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{kb.size}</td>
                      <td className="px-3 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{kb.lastUpdated}</td>
                      <td className="px-3 py-4 whitespace-nowrap text-sm">
                        <StatusBadge status={kb.status} />
                      </td>
                      <td className="px-3 py-4 whitespace-nowrap text-sm">
                        <div className="flex space-x-2">
                          <button 
                            className="text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-300"
                            onClick={() => setSelectedKB(kb.name)}
                          >
                            Browse
                          </button>
                          <button className="text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-300">Update</button>
                          <button className="text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300">Delete</button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {selectedKB && (
            <Card title={`${selectedKB} - Documents`}>
              <div className="flex justify-between items-center mb-4">
                <div className="flex items-center space-x-2">
                  <div className="relative">
                    <input
                      type="text"
                      className="pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white rounded-md"
                      placeholder="Search documents..."
                    />
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <Search size={18} className="text-gray-400" />
                    </div>
                  </div>
                  <button className="p-2 border border-gray-300 dark:border-gray-700 rounded-md text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300">
                    <Filter size={18} />
                  </button>
                </div>
                <button 
                  className="px-3 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 flex items-center"
                >
                  <Plus size={16} className="mr-1" />
                  Add Document
                </button>
              </div>

              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                  <thead>
                    <tr>
                      <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Title</th>
                      <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Type</th>
                      <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Size (MB)</th>
                      <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Uploaded</th>
                      <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Last Accessed</th>
                      <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                    {documents.map((doc, index) => (
                      <tr key={doc.id} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                        <td className="px-3 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">{doc.title}</td>
                        <td className="px-3 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{doc.type}</td>
                        <td className="px-3 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{doc.size}</td>
                        <td className="px-3 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{new Date(doc.uploadedAt).toLocaleDateString()}</td>
                        <td className="px-3 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{new Date(doc.lastAccessed).toLocaleDateString()}</td>
                        <td className="px-3 py-4 whitespace-nowrap text-sm">
                          <div className="flex space-x-2">
                            <button className="text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-300">
                              <Eye size={16} />
                            </button>
                            <button className="text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-300">
                              <Edit size={16} />
                            </button>
                            <button className="text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300">
                              <Trash2 size={16} />
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))
