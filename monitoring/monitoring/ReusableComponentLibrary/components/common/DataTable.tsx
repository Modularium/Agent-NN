// DataTable.tsx - Reusable data table component with sorting, filtering, and pagination
import React, { useState, useEffect, useMemo } from 'react';
import { ChevronUp, ChevronDown, Search, Filter, ArrowLeft, ArrowRight, RefreshCw, Download, Trash2, Settings, Plus, Check } from 'lucide-react';
import { useTheme } from '../../context/ThemeContext';

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
  width?: string;
  minWidth?: string;
  maxWidth?: string;
  hide?: boolean;
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
  onExport?: () => void;
  onDelete?: (selectedRows: T[]) => void;
  onAdd?: () => void;
  searchPlaceholder?: string;
  selectable?: boolean;
  className?: string;
  rowClassName?: (row: T) => string;
  headerClassName?: string;
  bodyClassName?: string;
  actions?: React.ReactNode;
  title?: string;
  subtitle?: string;
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
  onExport,
  onDelete,
  onAdd,
  searchPlaceholder = 'Search...',
  selectable = false,
  className = '',
  rowClassName,
  headerClassName = '',
  bodyClassName = '',
  actions,
  title,
  subtitle,
}: DataTableProps<T>) {
  const { themeMode } = useTheme();
  const [sortBy, setSortBy] = useState<{ id: string; direction: SortDirection }>(
    initialSortBy || { id: '', direction: null }
  );
  const [currentPage, setCurrentPage] = useState(1);
  const [searchTerm, setSearchTerm] = useState('');
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [activeFilter, setActiveFilter] = useState<string | null>(null);
  const [filterPosition, setFilterPosition] = useState({ top: 0, left: 0 });
  const [selectedRows, setSelectedRows] = useState<T[]>([]);
  const [selectAll, setSelectAll] = useState(false);

  // Reset to first page when search term changes
  useEffect(() => {
    setCurrentPage(1);
  }, [searchTerm, filters]);

  // Handle select all
  useEffect(() => {
    if (selectAll) {
      setSelectedRows(filteredData);
    } else if (selectedRows.length === filteredData.length) {
      setSelectedRows([]);
    }
  }, [selectAll]);

  // Search and filter data
  const filteredData = useMemo(() => {
    return data.filter(row => {
      // Apply search
      if (searchTerm) {
        const searchString = searchTerm.toLowerCase();
        const searchMatch = columns.some(column => {
          if (column.hide) return false;
          
          const value = column.accessor(row);
          return value !== null && value !== undefined &&
            String(value).toLowerCase().includes(searchString);
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
        
        const stringValue = String(value).toLowerCase();
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
      const aString = String(aValue);
      const bString = String(bValue);
      
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
    if (!onRefresh || isRefreshing) return;
    
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

  const goToPage = (page: number) => {
    setCurrentPage(Math.min(Math.max(1, page), totalPages));
  };

  // Handle row selection
  const isRowSelected = (row: T) => {
    return selectedRows.some(selectedRow => 
      rowKeyAccessor(selectedRow) === rowKeyAccessor(row)
    );
  };

  const toggleRowSelection = (row: T) => {
    if (isRowSelected(row)) {
      setSelectedRows(selectedRows.filter(selectedRow => 
        rowKeyAccessor(selectedRow) !== rowKeyAccessor(row)
      ));
    } else {
      setSelectedRows([...selectedRows, row]);
    }
  };

  const toggleSelectAll = () => {
    setSelectAll(!selectAll);
    if (!selectAll) {
      setSelectedRows(filteredData);
    } else {
      setSelectedRows([]);
    }
  };

  // Handle filter click
  const handleFilterClick = (columnId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const column = columns.find(col => col.id === columnId);
    if (!column || column.filterable === false) return;
    
    // Toggle filter popup
    if (activeFilter === columnId) {
      setActiveFilter(null);
    } else {
      setActiveFilter(columnId);
      
      // Position the filter popup
      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
      setFilterPosition({
        top: rect.bottom + window.scrollY,
        left: rect.left + window.scrollX
      });
    }
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
    setActiveFilter(null);
  };

  // Clear all filters
  const handleClearFilters = () => {
    setFilters({});
    setSearchTerm('');
  };

  // Handle bulk actions
  const handleDelete = () => {
    if (onDelete && selectedRows.length > 0) {
      onDelete(selectedRows);
      setSelectedRows([]);
    }
  };

  // Format cell value
  const formatCellValue = (value: any): React.ReactNode => {
    if (value === null || value === undefined) {
      return <span className="text-gray-400 dark:text-gray-600">—</span>;
    }
    
    if (typeof value === 'boolean') {
      return value ? 'Yes' : 'No';
    }
    
    if (value instanceof Date) {
      return value.toLocaleString();
    }
    
    return String(value);
  };

  // Generate page numbers for pagination
  const generatePageNumbers = () => {
    const pages = [];
    const maxVisiblePages = 5;
    
    if (totalPages <= maxVisiblePages) {
      // Show all pages if there are few
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
    } else {
      // Show a subset of pages with ellipsis
      if (currentPage <= 3) {
        // Near the start
        for (let i = 1; i <= 3; i++) {
          pages.push(i);
        }
        pages.push('ellipsis');
        pages.push(totalPages);
      } else if (currentPage >= totalPages - 2) {
        // Near the end
        pages.push(1);
        pages.push('ellipsis');
        for (let i = totalPages - 2; i <= totalPages; i++) {
          pages.push(i);
        }
      } else {
        // In the middle
        pages.push(1);
        pages.push('ellipsis');
        pages.push(currentPage - 1);
        pages.push(currentPage);
        pages.push(currentPage + 1);
        pages.push('ellipsis');
        pages.push(totalPages);
      }
    }
    
    return pages;
  };

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden ${className}`}>
      {/* Table header with search and actions */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
          {/* Title */}
          {(title || subtitle) && (
            <div>
              {title && <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{title}</h3>}
              {subtitle && <p className="text-sm text-gray-500 dark:text-gray-400">{subtitle}</p>}
            </div>
          )}
          
          {/* Search and actions */}
          <div className="flex flex-wrap items-center gap-2">
            {showSearch && (
              <div className="relative">
                <input
                  type="text"
                  placeholder={searchPlaceholder}
                  className="pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-white rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  value={searchTerm}
                  onChange={e => setSearchTerm(e.target.value)}
                />
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search size={18} className="text-gray-400" />
                </div>
              </div>
            )}
            
            {/* Filter indicators */}
            {Object.keys(filters).length > 0 && (
              <div className="flex items-center gap-2">
                <div className="text-xs text-gray-500 dark:text-gray-400">Filters:</div>
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
                          className="ml-1 text-indigo-700 dark:text-indigo-300 hover:text-indigo-900 dark:hover:text-indigo-100"
                          onClick={() => handleFilterApply(columnId, '')}
                        >
                          ×
                        </button>
                      </div>
                    );
                  })}
                  <button
                    className="text-xs text-indigo-600 dark:text-indigo-400 hover:underline"
                    onClick={handleClearFilters}
                  >
                    Clear all
                  </button>
                </div>
              </div>
            )}
            
            <div className="flex ml-auto gap-2">
              {/* Refresh button */}
              {onRefresh && (
                <button
                  className="p-2 rounded text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                  onClick={handleRefresh}
                  disabled={isRefreshing}
                >
                  <RefreshCw size={18} className={isRefreshing ? "animate-spin" : ""} />
                </button>
              )}
              
              {/* Export button */}
              {onExport && (
                <button
                  className="p-2 rounded text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                  onClick={onExport}
                >
                  <Download size={18} />
                </button>
              )}
              
              {/* Delete button - active when rows are selected */}
              {onDelete && selectedRows.length > 0 && (
                <button
                  className="p-2 rounded text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 hover:bg-red-50 dark:hover:bg-red-900/20"
                  onClick={handleDelete}
                >
                  <Trash2 size={18} />
                </button>
              )}
              
              {/* Add button */}
              {onAdd && (
                <button
                  className="flex items-center space-x-1 px-3 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 dark:bg-indigo-700 dark:hover:bg-indigo-600"
                  onClick={onAdd}
                >
                  <Plus size={18} />
                  <span>Add</span>
                </button>
              )}
              
              {/* Custom actions */}
              {actions}
            </div>
          </div>
        </div>
      </div>
      
      {/* Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead className={`bg-gray-50 dark:bg-gray-800 ${headerClassName}`}>
            <tr>
              {/* Selection checkbox */}
              {selectable && (
                <th className="px-3 py-3 text-left">
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 dark:border-gray-600 dark:bg-gray-700"
                      checked={selectAll}
                      onChange={toggleSelectAll}
                    />
                  </div>
                </th>
              )}
              
              {/* Column headers */}
              {columns.filter(col => !col.hide).map(column => (
                <th 
                  key={column.id}
                  className={`px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider ${column.headerClassName || ''}`}
                  style={{
                    width: column.width,
                    minWidth: column.minWidth,
                    maxWidth: column.maxWidth
                  }}
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
                    
                    {/* Filter button */}
                    {column.filterable !== false && (
                      <button
                        className={`p-1 rounded filter-button ${
                          activeFilter === column.id 
                            ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300'
                            : filters[column.id] 
                              ? 'bg-indigo-50 text-indigo-700 dark:bg-indigo-900/20 dark:text-indigo-300'
                              : 'text-gray-400 dark:text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
                        }`}
                        onClick={(e) => handleFilterClick(column.id, e)}
                      >
                        <Filter size={14} />
                      </button>
                    )}
                  </div>
                  
                  {/* Filter popup */}
                  {activeFilter === column.id && (
                    <div 
                      className="filter-popup absolute bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md shadow-lg z-10 p-3 mt-1"
                      style={{
                        top: filterPosition.top,
                        left: filterPosition.left
                      }}
                      onClick={e => e.stopPropagation()}
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
                              setActiveFilter(null);
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
          <tbody className={`bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700 ${bodyClassName}`}>
            {isLoading ? (
              <tr>
                <td
                  colSpan={columns.filter(col => !col.hide).length + (selectable ? 1 : 0)}
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
                  colSpan={columns.filter(col => !col.hide).length + (selectable ? 1 : 0)}
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
                  } ${isRowSelected(row) ? 'bg-indigo-50 dark:bg-indigo-900/20' : ''} ${rowClassName ? rowClassName(row) : ''}`}
                  onClick={() => onRowClick && onRowClick(row)}
                >
                  {/* Selection checkbox */}
                  {selectable && (
                    <td className="px-3 py-4" onClick={e => e.stopPropagation()}>
                      <div className="flex items-center">
                        <input
                          type="checkbox"
                          className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 dark:border-gray-600 dark:bg-gray-700"
                          checked={isRowSelected(row)}
                          onChange={() => toggleRowSelection(row)}
                        />
                      </div>
                    </td>
                  )}
                  
                  {/* Row cells */}
                  {columns.filter(col => !col.hide).map(column => (
                    <td
                      key={column.id}
                      className={`px-3 py-4 whitespace-nowrap text-sm ${column.className || ''}`}
                    >
                      {column.cell
                        ? column.cell(column.accessor(row), row)
                        : formatCellValue(column.accessor(row))}
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
                {generatePageNumbers().map((page, index) => (
                  <React.Fragment key={index}>
                    {page === 'ellipsis' ? (
                      <span className="relative inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-sm font-medium text-gray-700 dark:text-gray-300">
                        ...
                      </span>
                    ) : (
                      <button
                        onClick={() => goToPage(page as number)}
                        className={`relative inline-flex items-center px-4 py-2 border text-sm font-medium ${
                          currentPage === page
                            ? 'z-10 bg-indigo-50 dark:bg-indigo-900/20 border-indigo-500 dark:border-indigo-500 text-indigo-600 dark:text-indigo-300'
                            : 'bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 text-gray-500 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700'
                        }`}
                      >
                        {page}
                      </button>
                    )}
                  </React.Fragment>
                ))}
                
                <button
                  onClick={handleNextPage}
                  disabled={currentPage === totalPages}
                  className={`relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 dark:border-gray-600 text-sm font-medium ${
                    currentPage === totalPages
                      ? 'text-gray-300 dark:text-gray-600 bg-gray-50 dark:bg-gray-800'
                      : 'text-gray-500 dark:text-gray-400 bg-white dark:bg-
