// src/components/common/ExportButton.tsx
import React, { useState } from 'react';
import { Download, FileText, FileSpreadsheet, FilePdf, ChevronDown, Check } from 'lucide-react';
import { 
  exportAsJSON, 
  exportAsCSV, 
  exportAsPDF, 
  exportAsExcel,
  generateFilename 
} from '../../utils/exportUtils';
import { useNotification } from './NotificationSystem';

export type ExportFormat = 'json' | 'csv' | 'excel' | 'pdf';

interface ExportButtonProps {
  data: any;
  filename?: string;
  formats?: ExportFormat[];
  onExport?: (format: ExportFormat) => void;
  disabled?: boolean;
  className?: string;
  label?: string;
  showIcon?: boolean;
  variant?: 'default' | 'primary' | 'outline' | 'minimal';
  size?: 'sm' | 'md' | 'lg';
  csvHeaders?: string[];
}

const ExportButton: React.FC<ExportButtonProps> = ({
  data,
  filename = 'export',
  formats = ['json', 'csv'],
  onExport,
  disabled = false,
  className = '',
  label = 'Export',
  showIcon = true,
  variant = 'default',
  size = 'md',
  csvHeaders
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const { addNotification } = useNotification();
  
  // Handle dropdown toggle
  const toggleDropdown = () => {
    if (!disabled) {
      setIsOpen(!isOpen);
    }
  };
  
  // Handle export
  const handleExport = (format: ExportFormat) => {
    try {
      if (onExport) {
        onExport(format);
      }
      
      // Generate filename with extension
      const fullFilename = generateFilename(filename, format);
      
      // Export based on format
      switch (format) {
        case 'json':
          exportAsJSON(data, fullFilename);
          break;
        case 'csv':
          if (Array.isArray(data)) {
            exportAsCSV(data, fullFilename, csvHeaders);
          } else {
            throw new Error('Data must be an array for CSV export');
          }
          break;
        case 'excel':
          if (Array.isArray(data)) {
            exportAsExcel(data, fullFilename.replace('.excel', '.xlsx'));
          } else {
            throw new Error('Data must be an array for Excel export');
          }
          break;
        case 'pdf':
          exportAsPDF(data, fullFilename);
          break;
      }
      
      // Close dropdown
      setIsOpen(false);
      
      // Show success notification
      addNotification({
        type: 'success',
        title: 'Export Successful',
        message: `Data exported as ${format.toUpperCase()}`,
        duration: 3000
      });
    } catch (error) {
      console.error('Export error:', error);
      
      // Show error notification
      addNotification({
        type: 'error',
        title: 'Export Failed',
        message: error instanceof Error ? error.message : 'An error occurred during export',
        duration: 5000
      });
    }
  };
  
  // Button variants
  const variantClasses = {
    default: 'bg-white text-gray-800 border border-gray-300 hover:bg-gray-50 dark:bg-gray-800 dark:text-gray-200 dark:border-gray-700 dark:hover:bg-gray-700',
    primary: 'bg-indigo-600 text-white hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600',
    outline: 'bg-transparent text-indigo-600 border border-indigo-600 hover:bg-indigo-50 dark:text-indigo-400 dark:border-indigo-400 dark:hover:bg-indigo-900/20',
    minimal: 'bg-transparent text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800'
  };
  
  // Button sizes
  const sizeClasses = {
    sm: 'text-xs px-2 py-1',
    md: 'text-sm px-3 py-2',
    lg: 'text-base px-4 py-2'
  };
  
  // Format options
  const
