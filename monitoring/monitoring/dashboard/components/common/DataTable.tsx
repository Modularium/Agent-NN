// monitoring/dashboard/components/common/DataTable.tsx
import React, { useState, useEffect, useMemo } from 'react';
import { ChevronUp, ChevronDown, Search, Filter, ArrowLeft, ArrowRight, RefreshCw } from 'lucide-react';

export type SortDirection = 'asc' | 'desc' | null;

export interface Column<T> {
  id: string;
  header: string;
  accessor: (row: T) => any;
  sortable?: boolean;
  filterable?: boolean;
  cell?: (value: any, row: T) => React.ReactNode;
  className?: string;
  headerClassName?: string;
}

interface DataTableProps<T> {
  data: T[];
  columns: Column<T>[];
  initialSortBy?: { id: string; direction: SortDirection };
  rowKeyAccessor?: (row: T) => string;
  onRowClick?: (row: T) => void;
  isLoading?: boolean;
  emptyMessage?: string;
  showPagination?: boolean;
  itemsPerPage?: number;
  showSearch?: boolean;
  onRefresh?: () => Promise<void>;
  className?: string;
  rowClassName?: (row: T) => string;
  actions?: React.ReactNode;
}

function DataTable<T>({
  data,
  columns,
  initialSortBy,
  rowKeyAccessor = (row: T) => JSON.stringify(row),
  onRowClick,
  isLoading = false,
  emptyMessage = 'No data available',
  showPagination = true,
  itemsPerPage = 10,
  showSearch = true,
  onRefresh,
  className = '',
  rowClassName,
  actions
}: DataTableProps<T>) {
  const [sortBy, setSortBy] = useState<{ id: string; direction: SortDirection }>(
    initialSortBy || { id: '', direction: null }
  );
  const [currentPage, setCurrentPage] = useState(1);
  const [searchTerm, setSearchTerm] = useState('');
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [activeColumn, setActiveColumn] = useState<string | null>(null);
  const [filterPopupPosition, setFilterPopupPosition] = useState({ top: 0, left: 0 });

  // Reset to first page when search term changes
  useEffect(() => {
    setCurrentPage(1);
  }, [searchTerm, filters]);

  // Search and filter data
  const filteredData = useMemo(() => {
    return data.filter(row => {
      // Apply search
      if (searchTerm) {
        const searchString = searchTerm.toLowerCase();
        const searchMatch = columns.some(column => {
          const value = column.accessor(row);
          return value !== null && value !== undefined &&
            value.toString().toLowerCase().includes(searchString);
        });
        
        if (!searchMatch) return false;
      }
      
      // Apply filters
      for (const [columnId, filterValue] of Object.entries(filters)) {
        if (!filterValue) continue;
        
        const column = columns.find(col => col.id === columnId);
        if (!column) continue;
        
        const value = column.accessor(row);
        if (value === null || value === undefined) return false;
        
        const stringValue = value.toString().toLowerCase();
        if (!stringValue.includes(filterValue.toLowerCase())) return false;
      }
      
      return true;
    });
  }, [data, columns, searchTerm, filters]);

  // Sort data
  const sortedData = useMemo(() => {
    if (!sortBy.id || sortBy.direction === null) return filteredData;
    
    const column = columns.find(col => col.id === sortBy.id);
    if (!column) return filteredData;
    
    return [...filteredData].sort((a, b) => {
      const aValue = column.accessor(a);
      const bValue = column.accessor(b);
      
      if (aValue === bValue) return 0;
      
      // Handle null/undefined values
      if (aValue === null || aValue === undefined) return sortBy.direction === 'asc' ? -1 : 1;
      if (bValue === null || bValue === undefined) return sortBy.direction === 'asc' ? 1 : -1;
      
      // Sort numbers
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return sortBy.direction === 'asc' ? aValue - bValue : bValue - aValue;
      }
      
      // Sort dates
      if (aValue instanceof Date && bValue instanceof Date) {
        return sortBy.direction === 'asc' 
          ? aValue.getTime() - bValue.getTime() 
          : bValue.getTime() - aValue.getTime();
      }
      
      // Sort strings
      const aString = aValue.toString();
      const bString = bValue.toString();
      
      return sortBy.direction === 'asc'
        ? aString.localeCompare(bString)
        : bString.localeCompare(aString);
    });
  }, [filteredData, sortBy, columns]);

  // Paginate data
  const paginatedData = useMemo(() => {
    if (!showPagination) return sortedData;
    
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    
    return sortedData.slice(startIndex, endIndex);
  }, [sortedData, currentPage, itemsPerPage, showPagination]);

  // Calculate total pages
  const totalPages = useMemo(() => {
    if (!showPagination) return 1;
    return Math.max(1, Math.ceil(sortedData.length / itemsPerPage));
  }, [sortedData, itemsPerPage, showPagination]);

  // Handle sort
  const handleSort = (columnId: string) => {
    const column = columns.find(col => col.id === columnId);
    if (!column || column.sortable === false) return;
    
    if (sortBy.id === columnId) {
      // Cycle through: asc -> desc -> null
      let newDirection: SortDirection = null;
      if (sortBy.direction === null) newDirection = 'asc';
      else if (sortBy.direction === 'asc') newDirection = 'desc';
      else newDirection = null;
      
      setSortBy({ id: columnId, direction: newDirection });
    } else {
      setSortBy({ id: columnId, direction: 'asc' });
    }
  };

  // Handle refresh
  const handleRefresh = async () => {
    if (!onRefresh) return;
    
    setIsRefreshing(true);
    try {
      await onRefresh();
    } finally {
      setIsRefreshing(false);
    }
  };

  // Handle page change
  const handlePreviousPage = () => {
    setCurrentPage(prev => Math.max(1, prev - 1));
  };

  const handleNextPage = () => {
    setCurrentPage(prev => Math.min(totalPages, prev + 1));
  };

  // Show/hide filter popup
  const handleFilterClick = (columnId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const column = columns.find(col => col.id === columnId);
    if (!column || column.filterable === false) return;
    
    // If already active, hide the popup
    if (activeColumn === columnId) {
      setActiveColumn(null);
      return;
    }
    
    // Show popup for this column
    setActiveColumn(columnId);
    
    // Calculate popup position
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    setFilterPopupPosition({
      top: rect.bottom + window.scrollY,
      left: rect.left + window.scrollX
    });
  };

  // Apply filter
  const handleFilterApply = (columnId: string, value: string) => {
    if (value) {
      setFilters(prev => ({ ...prev, [columnId]: value }));
    } else {
      setFilters(prev => {
        const newFilters = { ...prev };
        delete newFilters[columnId];
        return newFilters;
      });
    }
    setActiveColumn(null);
  };

  // Handle outside click to close filter popup
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (!activeColumn) return;
      
      const isOutside = !(e.target as HTMLElement).closest('.filter-popup');
      const isNotFilterButton = !(e.target as HTMLElement).closest('.filter-button');
      
      if (isOutside && isNotFilterButton) {
        setActiveColumn(null);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [activeColumn]);

  return (
    <div className={`overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700 ${className}`}>
      {/* Table header with search and actions */}
      <div className="bg-white dark:bg-gray-800 p-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
        {showSearch ? (
          <div className="relative">
            <input
              type="text"
              placeholder="Search..."
              className="pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white rounded-md focus:ring-indigo-500 focus:border-indigo-500"
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
            />
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search size={18} className="text-gray-400" />
            </div>
          </div>
        ) : (
          <div></div> // Empty div to maintain flex layout
        )}
        
        <div className="flex items-center space-x-2">
          {/* Applied filters display */}
          {Object.keys(filters).length > 0 && (
            <div className="flex items-center mr-2">
              <span className="text-xs text-gray-500 dark:text-gray-400 mr-2">Filters:</span>
              <div className="flex flex-wrap gap-1">
                {Object.entries(filters).map(([columnId, value]) => {
                  const column = columns.find(col => col.id === columnId);
                  return (
                    <div 
                      key={columnId}
                      className="flex items-center bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 text-xs px-2 py-1 rounded"
                    >
                      <span>{column?.header || columnId}: {value}</span>
                      <button
                        className="ml-1"
                        onClick={() => handleFilterApply(columnId, '')}
                      >
                        ×
                      </button>
                    </div>
                  );
                })}
                <button
                  className="text-xs text-indigo-600 dark:text-indigo-400 hover:underline"
                  onClick={() => setFilters({})}
                >
                  Clear all
                </button>
              </div>
            </div>
          )}
          
          {/* Refresh button */}
          {onRefresh && (
            <button
              className="p-2 rounded text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
              onClick={handleRefresh}
              disabled={isRefreshing}
            >
              <RefreshCw size={16} className={isRefreshing ? "animate-spin" : ""} />
            </button>
          )}
          
          {/* Custom actions */}
          {actions}
        </div>
      </div>
      
      {/* Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead className="bg-gray-50 dark:bg-gray-800">
            <tr>
              {columns.map(column => (
                <th
                  key={column.id}
                  scope="col"
                  className={`px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider ${column.headerClassName || ''}`}
                >
                  <div className="flex items-center space-x-1">
                    <button
                      className={`flex items-center font-medium ${column.sortable === false ? 'cursor-default' : 'cursor-pointer hover:text-gray-700 dark:hover:text-gray-300'}`}
                      onClick={() => handleSort(column.id)}
                      disabled={column.sortable === false}
                    >
                      <span>{column.header}</span>
                      {column.sortable !== false && (
                        <span className="ml-1">
                          {sortBy.id === column.id ? (
                            sortBy.direction === 'asc' ? (
                              <ChevronUp size={14} />
                            ) : sortBy.direction === 'desc' ? (
                              <ChevronDown size={14} />
                            ) : null
                          ) : null}
                        </span>
                      )}
                    </button>
                    
                    {column.filterable !== false && (
                      <button
                        className={`p-1 rounded filter-button ${
                          activeColumn === column.id 
                            ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300'
                            : filters[column.id] 
                              ? 'bg-indigo-50 text-indigo-700 dark:bg-indigo-900/20 dark:text-indigo-300'
                              : 'text-gray-400 dark:text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
                        }`}
                        onClick={e => handleFilterClick(column.id, e)}
                      >
                        <Filter size={14} />
                      </button>
                    )}
                  </div>
                  
                  {/* Filter popup */}
                  {activeColumn === column.id && (
                    <div
                      className="filter-popup absolute bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md shadow-lg z-10 p-3 mt-1"
                      style={{
                        top: filterPopupPosition.top,
                        left: filterPopupPosition.left
                      }}
                    >
                      <div className="text-sm text-gray-900 dark:text-gray-100 mb-2">
                        Filter by {column.header}
                      </div>
                      <div className="flex">
                        <input
                          type="text"
                          className="form-input rounded-md border-gray-300 dark:border-gray-700 dark:bg-gray-700 dark:text-white text-sm mr-2"
                          placeholder="Filter value..."
                          defaultValue={filters[column.id] || ''}
                          autoFocus
                          onKeyDown={e => {
                            if (e.key === 'Enter') {
                              handleFilterApply(column.id, e.currentTarget.value);
                            } else if (e.key === 'Escape') {
                              setActiveColumn(null);
                            }
                          }}
                        />
                        <button
                          className="px-2 py-1 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600 text-sm"
                          onClick={e => {
                            const input = (e.currentTarget.previousSibling as HTMLInputElement);
                            handleFilterApply(column.id, input.value);
                          }}
                        >
                          Apply
                        </button>
                      </div>
                    </div>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
            {isLoading ? (
              <tr>
                <td
                  colSpan={columns.length}
                  className="px-3 py-4 text-center text-gray-500 dark:text-gray-400"
                >
                  <div className="flex justify-center items-center">
                    <RefreshCw size={20} className="animate-spin mr-2" />
                    <span>Loading...</span>
                  </div>
                </td>
              </tr>
            ) : paginatedData.length === 0 ? (
              <tr>
                <td
                  colSpan={columns.length}
                  className="px-3 py-4 text-center text-gray-500 dark:text-gray-400"
                >
                  {emptyMessage}
                </td>
              </tr>
            ) : (
              paginatedData.map(row => (
                <tr
                  key={rowKeyAccessor(row)}
                  className={`${
                    onRowClick ? 'cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700' : ''
                  } ${rowClassName ? rowClassName(row) : ''}`}
                  onClick={() => onRowClick && onRowClick(row)}
                >
                  {columns.map(column => (
                    <td
                      key={column.id}
                      className={`px-3 py-4 whitespace-nowrap text-sm ${column.className || ''}`}
                    >
                      {column.cell
                        ? column.cell(column.accessor(row), row)
                        : renderCellValue(column.accessor(row))}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
      
      {/* Pagination */}
      {showPagination && totalPages > 1 && (
        <div className="bg-white dark:bg-gray-800 px-4 py-3 flex items-center justify-between border-t border-gray-200 dark:border-gray-700">
          <div className="flex-1 flex justify-between sm:hidden">
            <button
              onClick={handlePreviousPage}
              disabled={currentPage === 1}
              className={`relative inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 text-sm font-medium rounded-md ${
                currentPage === 1
                  ? 'text-gray-300 dark:text-gray-600 bg-gray-50 dark:bg-gray-800'
                  : 'text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
            >
              Previous
            </button>
            <button
              onClick={handleNextPage}
              disabled={currentPage === totalPages}
              className={`ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 text-sm font-medium rounded-md ${
                currentPage === totalPages
                  ? 'text-gray-300 dark:text-gray-600 bg-gray-50 dark:bg-gray-800'
                  : 'text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
            >
              Next
            </button>
          </div>
          <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
            <div>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                Showing <span className="font-medium">{(currentPage - 1) * itemsPerPage + 1}</span> to{' '}
                <span className="font-medium">
                  {Math.min(currentPage * itemsPerPage, sortedData.length)}
                </span>{' '}
                of <span className="font-medium">{sortedData.length}</span> results
              </p>
            </div>
            <div>
              <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px" aria-label="Pagination">
                <button
                  onClick={handlePreviousPage}
                  disabled={currentPage === 1}
                  className={`relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 dark:border-gray-600 text-sm font-medium ${
                    currentPage === 1
                      ? 'text-gray-300 dark:text-gray-600 bg-gray-50 dark:bg-gray-800'
                      : 'text-gray-500 dark:text-gray-400 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700'
                  }`}
                >
                  <span className="sr-only">Previous</span>
                  <ArrowLeft size={16} />
                </button>
                
                {/* Page numbers */}
                {renderPageNumbers()}
                
                <button
                  onClick={handleNextPage}
                  disabled={currentPage === totalPages}
                  className={`relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 dark:border-gray-600 text-sm font-medium ${
                    currentPage === totalPages
                      ? 'text-gray-300 dark:text-gray-600 bg-gray-50 dark:bg-gray-800'
                      : 'text-gray-500 dark:text-gray-400 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700'
                  }`}
                >
                  <span className="sr-only">Next</span>
                  <ArrowRight size={16} />
                </button>
              </nav>
            </div>
          </div>
        </div>
      )}
    </div>
  );
  
  // Helper function to render cell values
  function renderCellValue(value: any): React.ReactNode {
    if (value === null || value === undefined) {
      return <span className="text-gray-400 dark:text-gray-600">-</span>;
    }
    
    if (typeof value === 'boolean') {
      return value ? 'Yes' : 'No';
    }
    
    if (value instanceof Date) {
      return value.toLocaleString();
    }
    
    return value.toString();
  }
  
  // Helper function to render page numbers
  function renderPageNumbers() {
    const pageNumbers = [];
    
    // For small number of pages, show all
    if (totalPages <= 7) {
      for (let i = 1; i <= totalPages; i++) {
        pageNumbers.push(renderPageButton(i));
      }
    } else {
      // For many pages, show ellipsis
      if (currentPage <= 3) {
        // Near start
        for (let i = 1; i <= 5; i++) {
          pageNumbers.push(renderPageButton(i));
        }
        pageNumbers.push(renderEllipsis());
        pageNumbers.push(renderPageButton(totalPages));
      } else if (currentPage >= totalPages - 2) {
        // Near end
        pageNumbers.push(renderPageButton(1));
        pageNumbers.push(renderEllipsis());
        for (let i = totalPages - 4; i <= totalPages; i++) {
          pageNumbers.push(renderPageButton(i));
        }
      } else {
        // In middle
        pageNumbers.push(renderPageButton(1));
        pageNumbers.push(renderEllipsis());
        for (let i = currentPage - 1; i <= currentPage + 1; i++) {
          pageNumbers.push(renderPageButton(i));
        }
        pageNumbers.push(renderEllipsis());
        pageNumbers.push(renderPageButton(totalPages));
      }
    }
    
    return pageNumbers;
  }
  
  // Render a single page button
  function renderPageButton(pageNumber: number) {
    return (
      <button
        key={pageNumber}
        onClick={() => setCurrentPage(pageNumber)}
        aria-current={pageNumber === currentPage ? 'page' : undefined}
        className={`relative inline-flex items-center px-4 py-2 border text-sm font-medium ${
          pageNumber === currentPage
            ? 'z-10 bg-indigo-50 dark:bg-indigo-900/20 border-indigo-500 dark:border-indigo-500 text-indigo-600 dark:text-indigo-300'
            : 'bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 text-gray-500 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700'
        }`}
      >
        {pageNumber}
      </button>
    );
  }
  
  // Render ellipsis
  function renderEllipsis() {
    return (
      <span
        key="ellipsis"
        className="relative inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-sm font-medium text-gray-700 dark:text-gray-300"
      >
        ...
      </span>
    );
  }
}
