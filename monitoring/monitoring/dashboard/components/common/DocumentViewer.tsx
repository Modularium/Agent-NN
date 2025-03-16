// monitoring/dashboard/components/common/DocumentViewer.tsx
import React, { useState, useEffect, useRef } from 'react';
import { 
  ChevronLeft, 
  ChevronRight, 
  Search, 
  ZoomIn, 
  ZoomOut, 
  RotateCw, 
  Download, 
  Printer, 
  X,
  Maximize,
  Minimize,
  ArrowLeft,
  ArrowRight,
  Plus,
  Minus
} from 'lucide-react';

// Document types the viewer can handle
export type DocumentType = 'pdf' | 'image' | 'text' | 'json' | 'csv' | 'html' | 'markdown';

export interface DocumentViewerProps {
  url: string;
  type?: DocumentType;
  title?: string;
  onClose?: () => void;
  className?: string;
  initialPage?: number;
  showToolbar?: boolean;
  showNavigation?: boolean;
  showThumbnails?: boolean;
  showSearch?: boolean;
  allowDownload?: boolean;
  allowPrint?: boolean;
  allowFullscreen?: boolean;
  maxHeight?: number | string;
  onDocumentLoaded?: (numPages: number) => void;
  onPageChange?: (pageNumber: number) => void;
  onError?: (error: Error) => void;
}

/**
 * A component for viewing different types of documents with controls
 */
const DocumentViewer: React.FC<DocumentViewerProps> = ({
  url,
  type,
  title,
  onClose,
  className = '',
  initialPage = 1,
  showToolbar = true,
  showNavigation = true,
  showThumbnails = false,
  showSearch = true,
  allowDownload = true,
  allowPrint = true,
  allowFullscreen = true,
  maxHeight,
  onDocumentLoaded,
  onPageChange,
  onError,
}) => {
  // Detect document type if not provided
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
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [currentSearchResult, setCurrentSearchResult] = useState(0);
  
  // References
  const viewerRef = useRef<HTMLDivElement>(null);
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  
  // When URL changes, reset viewer
  useEffect(() => {
    setIsLoading(true);
    setError(null);
    setCurrentPage(initialPage);
    setZoom(1);
    setRotation(0);
    setSearchQuery('');
    setSearchResults([]);
    setCurrentSearchResult(0);
    
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
    // If it's an iframe for PDF, we can print the iframe
    if (iframeRef.current && documentType === 'pdf') {
      const iframe = iframeRef.current;
      iframe.focus();
      iframe.contentWindow?.print();
    } else {
      // Otherwise create a new window with the content
      const printWindow = window.open(url, '_blank');
      if (printWindow) {
        printWindow.addEventListener('load', () => {
          printWindow.print();
        });
      }
    }
  };
  
  // Handle fullscreen
  const handleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };
  
  // Handle search
  const handleSearch = () => {
    if (!searchQuery) {
      setSearchResults([]);
      return;
    }
    
    // Mock search results
    const mockResults = [
      { page: 1, snippet: `...containing "${searchQuery}"...`, rect: { x: 100, y: 100, width: 200, height: 20 } },
      { page: 3, snippet: `...another match for "${searchQuery}"...`, rect: { x: 150, y: 200, width: 180, height: 20 } },
    ];
    
    setSearchResults(mockResults);
    
    if (mockResults.length > 0) {
      setCurrentSearchResult(0);
      handlePageChange(mockResults[0].page);
    }
  };
  
  // Navigate search results
  const navigateSearchResults = (direction: 'next' | 'prev') => {
    if (searchResults.length === 0) return;
    
    let newIndex;
    if (direction === 'next') {
      newIndex = (currentSearchResult + 1) % searchResults.length;
    } else {
      newIndex = (currentSearchResult - 1 + searchResults.length) % searchResults.length;
    }
    
    setCurrentSearchResult(newIndex);
    handlePageChange(searchResults[newIndex].page);
  };
  
  // Render document content based on type
  const renderDocumentContent = () => {
    if (isLoading) {
      return (
        <div className="flex items-center justify-center w-full h-full min-h-[300px]">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
        </div>
      );
    }
    
    if (error) {
      return (
        <div className="flex flex-col items-center justify-center w-full h-full min-h-[300px] p-4 text-red-600 dark:text-red-400">
          <div className="text-lg font-medium mb-2">Error loading document</div>
          <div className="text-sm">{error.message}</div>
        </div>
      );
    }
    
    switch (documentType) {
      case 'pdf':
        return (
          <iframe
            ref={iframeRef}
            src={`${url}#page=${currentPage}&zoom=${zoom * 100},0,0`}
            className="w-full h-full"
            style={{ 
              transform: `rotate(${rotation}deg)`,
              minHeight: '500px',
            }}
            title={title || 'PDF Document'}
          />
        );
        
      case 'image':
        return (
          <div className="flex
