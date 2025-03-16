// monitoring/dashboard/components/common/DocumentViewer.tsx
import React, { useState, useEffect } from 'react';
import { 
  XCircle, 
  Download, 
  Printer, 
  ZoomIn, 
  ZoomOut, 
  RotateCw, 
  ChevronLeft, 
  ChevronRight,
  Maximize,
  Minimize,
  Search,
  X
} from 'lucide-react';

// Document types the viewer can handle
export type DocumentType = 'pdf' | 'image' | 'text' | 'json' | 'csv' | 'html' | 'markdown';

interface DocumentViewerProps {
  url: string;
  type?: DocumentType;
  title?: string;
  onClose?: () => void;
  className?: string;
  initialPage?: number;
  showToolbar?: boolean;
  showNavigation?: boolean;
  allowDownload?: boolean;
  allowPrint?: boolean;
  allowFullscreen?: boolean;
  maxHeight?: number | string;
  onDocumentLoaded?: (numPages: number) => void;
  onPageChange?: (pageNumber: number) => void;
  onError?: (error: Error) => void;
}

const DocumentViewer: React.FC<DocumentViewerProps> = ({
  url,
  type,
  title,
  onClose,
  className = '',
  initialPage = 1,
  showToolbar = true,
  showNavigation = true,
  allowDownload = true,
  allowPrint = true,
  allowFullscreen = true,
  maxHeight = 600,
  onDocumentLoaded,
  onPageChange,
  onError
}) => {
  // Auto-detect document type if not provided
  const detectDocumentType = (url: string): DocumentType => {
    const extension = url.split('.').pop()?.toLowerCase();
    switch (extension) {
      case 'pdf':
        return 'pdf';
      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'gif':
      case 'svg':
        return 'image';
      case 'txt':
        return 'text';
      case 'json':
        return 'json';
      case 'csv':
        return 'csv';
      case 'html':
      case 'htm':
        return 'html';
      case 'md':
        return 'markdown';
      default:
        // Default to text
        return 'text';
    }
  };

  const documentType = type || detectDocumentType(url);
  
  // State for viewer
  const [currentPage, setCurrentPage] = useState(initialPage);
  const [numPages, setNumPages] = useState(1);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  
  // When URL changes, reset viewer
  useEffect(() => {
    setIsLoading(true);
    setError(null);
    setCurrentPage(initialPage);
    setZoom(1);
    setRotation(0);
    setSearchQuery('');
    
    // Simulate document loading
    const timer = setTimeout(() => {
      setIsLoading(false);
      
      // Mock number of pages based on document type
      const mockNumPages = documentType === 'pdf' ? 10 : 1;
      setNumPages(mockNumPages);
      
      if (onDocumentLoaded) {
        onDocumentLoaded(mockNumPages);
      }
    }, 1000);
    
    return () => clearTimeout(timer);
  }, [url, documentType, initialPage, onDocumentLoaded]);
  
  // Handle page change
  const handlePageChange = (newPage: number) => {
    if (newPage >= 1 && newPage <= numPages) {
      setCurrentPage(newPage);
      
      if (onPageChange) {
        onPageChange(newPage);
      }
    }
  };
  
  // Handle zoom
  const handleZoom = (factor: number) => {
    setZoom(prevZoom => {
      const newZoom = prevZoom + factor;
      return Math.max(0.25, Math.min(3, newZoom));
    });
  };
  
  // Handle rotation
  const handleRotate = () => {
    setRotation(prevRotation => (prevRotation + 90) % 360);
  };
  
  // Handle download
  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = url;
    link.download = title || url.split('/').pop() || 'document';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  // Handle print
  const handlePrint = () => {
    // Create a new window with the content for printing
    const printWindow = window.open(url, '_blank');
    if (printWindow) {
      printWindow.addEventListener('load', () => {
        printWindow.print();
      });
    }
  };
  
  // Handle fullscreen toggle
  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };
  
  // Render document content based on type
  const renderDocumentContent = () => {
    if (isLoading) {
      return (
        <div className="flex items-center justify-center w-full h-full min-h-[300px]">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-600"></div>
          <span className="ml-2 text-gray-600 dark:text-gray-300">Loading document...</span>
        </div>
      );
    }
    
    if (error) {
      return (
        <div className="flex flex-col items-center justify-center w-full h-full min-h-[300px] p-4 text-red-600 dark:text-red-400">
          <XCircle className="w-12 h-12 mb-2" />
          <div className="text-lg font-medium mb-2">Error loading document</div>
          <div className="text-sm">{error.message}</div>
        </div>
      );
    }
    
    switch (documentType) {
      case 'pdf':
        return (
          <iframe
            src={`${url}#page=${currentPage}&zoom=${zoom * 100},0,0`}
            className="w-full h-full"
            style={{ 
              transform: `rotate(${rotation}deg)`,
              maxHeight: isFullscreen ? 'none' : maxHeight,
              minHeight: isFullscreen ? '80vh' : 500
            }}
            title={title || 'PDF Document'}
          />
        );
        
      case 'image':
        return (
          <div className="flex items-center justify-center">
            <img 
              src={url} 
              alt={title || 'Document'} 
              style={{ 
                transform: `rotate(${rotation}deg) scale(${zoom})`,
                maxHeight: isFullscreen ? '80vh' : maxHeight
              }}
              className="max-w-full"
            />
          </div>
        );
        
      case 'text':
      case 'markdown':
        return (
          <iframe
            src={url}
            className="w-full h-full border-0"
            style={{ 
              maxHeight: isFullscreen ? 'none' : maxHeight,
              minHeight: isFullscreen ? '80vh' : 500
            }}
            title={title || 'Text Document'}
          />
        );
        
      case 'json':
        return (
          <iframe
            src={url}
            className="w-full h-full border-0"
            style={{ 
              maxHeight: isFullscreen ? 'none' : maxHeight,
              minHeight: isFullscreen ? '80vh' : 500
            }}
            title={title || 'JSON Document'}
          />
        );
        
      case 'csv':
        return (
          <iframe
            src={url}
            className="w-full h-full border-0"
            style={{ 
              maxHeight: isFullscreen ? 'none' : maxHeight,
              minHeight: isFullscreen ? '80vh' : 500
            }}
            title={title || 'CSV Document'}
          />
        );
        
      case 'html':
        return (
          <iframe
            src={url}
            className="w-full h-full border-0"
            style={{ 
              maxHeight: isFullscreen ? 'none' : maxHeight,
              minHeight: isFullscreen ? '80vh' : 500
            }}
            title={title || 'HTML Document'}
          />
        );
        
      default:
        return (
          <div className="flex flex-col items-center justify-center w-full h-full min-h-[300px] p-4 text-gray-600 dark:text-gray-400">
            <div className="text-lg font-medium mb-2">Unsupported document type</div>
            <div className="text-sm">Download the document to view it.</div>
          </div>
        );
    }
  };
  
  // Main container classes based on fullscreen state
  const containerClasses = isFullscreen
    ? 'fixed inset-0 z-50 bg-white dark:bg-gray-900 overflow-auto flex flex-col'
    : `bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden ${className}`;
  
  return (
    <div className={containerClasses}>
      {/* Header with title and close button */}
      <div className="flex justify-between items-center p-4 border-b border-gray-200 dark:border-gray-700">
        <h3 className="font-bold text-gray-900 dark:text-white truncate">
          {title || url.split('/').pop() || 'Document'}
        </h3>
        <div className="flex items-center space-x-2">
          {/* Search input for PDFs */}
          {documentType === 'pdf' && showToolbar && (
            <div className="relative max-w-xs">
              <input
                type="text"
                placeholder="Search..."
                className="pl-8 pr-3 py-1 text-sm border border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white rounded"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              <Search 
                size={14} 
                className="absolute left-2 top-1/2 transform -translate-y-1/2 text-gray-400"
              />
              {searchQuery && (
                <button 
                  className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                  onClick={() => setSearchQuery('')}
                >
                  <X size={14} />
                </button>
              )}
            </div>
          )}
          
          {/* Close button */}
          {onClose && (
            <button
              className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
              onClick={onClose}
              aria-label="Close"
            >
              <X size={20} />
            </button>
          )}
        </div>
      </div>
      
      {/* Toolbar */}
      {showToolbar && (
        <div className="flex justify-between items-center p-2 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
          <div className="flex items-center space-x-2">
            {/* Page navigation for PDFs */}
            {documentType === 'pdf' && showNavigation && (
              <div className="flex items-center">
                <button
                  className="p-1 rounded-md text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 disabled:opacity-50"
                  onClick={() => handlePageChange(currentPage - 1)}
                  disabled={currentPage <= 1}
                >
                  <ChevronLeft size={16} />
                </button>
                <span className="mx-2 text-sm text-gray-600 dark:text-gray-300">
                  Page {currentPage} of {numPages}
                </span>
                <button
                  className="p-1 rounded-md text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 disabled:opacity-50"
                  onClick={() => handlePageChange(currentPage + 1)}
                  disabled={currentPage >= numPages}
                >
                  <ChevronRight size={16} />
                </button>
              </div>
            )}
          </div>
          
          <div className="flex items-center space-x-2">
            {/* Zoom controls */}
            <button
              className="p-1 rounded-md text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
              onClick={() => handleZoom(-0.25)}
              aria-label="Zoom out"
            >
              <ZoomOut size={16} />
            </button>
            <span className="text-sm text-gray-600 dark:text-gray-300">
              {Math.round(zoom * 100)}%
            </span>
            <button
              className="p-1 rounded-md text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
              onClick={() => handleZoom(0.25)}
              aria-label="Zoom in"
            >
              <ZoomIn size={16} />
            </button>
            
            {/* Rotate control */}
            <button
              className="p-1 rounded-md text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
              onClick={handleRotate}
              aria-label="Rotate"
            >
              <RotateCw size={16} />
            </button>
            
            {/* Download button */}
            {allowDownload && (
              <button
                className="p-1 rounded-md text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
                onClick={handleDownload}
                aria-label="Download"
              >
                <Download size={16} />
              </button>
            )}
            
            {/* Print button */}
            {allowPrint && (
              <button
                className="p-1 rounded-md text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
                onClick={handlePrint}
                aria-label="Print"
              >
                <Printer size={16} />
              </button>
            )}
            
            {/* Fullscreen toggle button */}
            {allowFullscreen && (
              <button
                className="p-1 rounded-md text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
                onClick={toggleFullscreen}
                aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
              >
                {isFullscreen ? <Minimize size={16} /> : <Maximize size={16} />}
              </button>
            )}
          </div>
        </div>
      )}
      
      {/* Main document content */}
      <div className="flex-1 overflow-auto">
        {renderDocumentContent()}
      </div>
    </div>
  );
};

export default DocumentViewer;
