// src/utils/exportUtils.ts
/**
 * Utilities for exporting data from the dashboard in various formats
 */

// Export data as JSON file
export const exportAsJSON = <T>(data: T, filename: string = 'export.json'): void => {
  try {
    // Convert data to JSON string
    const jsonString = JSON.stringify(data, null, 2);
    
    // Create blob from JSON
    const blob = new Blob([jsonString], { type: 'application/json' });
    
    // Create download link
    downloadFile(blob, filename);
  } catch (error) {
    console.error('Error exporting JSON:', error);
    throw new Error('Failed to export data as JSON');
  }
};

// Export data as CSV file
export const exportAsCSV = <T extends Record<string, any>>(
  data: T[], 
  filename: string = 'export.csv',
  headers?: string[]
): void => {
  try {
    if (!data.length) {
      throw new Error('No data to export');
    }
    
    // Determine headers if not provided
    const csvHeaders = headers || Object.keys(data[0]);
    
    // Create CSV header row
    let csvContent = csvHeaders.join(',') + '\n';
    
    // Add data rows
    data.forEach(item => {
      const row = csvHeaders.map(header => {
        // Get the value for this header
        const value = item[header];
        
        // Handle different data types
        if (value === null || value === undefined) {
          return '';
        } else if (typeof value === 'string') {
          // Escape quotes and wrap in quotes if contains comma or quotes
          const escapedValue = value.replace(/"/g, '""');
          return /[",\n]/.test(value) ? `"${escapedValue}"` : escapedValue;
        } else if (typeof value === 'object') {
          // Convert objects to JSON strings and escape
          const json = JSON.stringify(value).replace(/"/g, '""');
          return `"${json}"`;
        } else {
          // Numbers, booleans, etc.
          return String(value);
        }
      }).join(',');
      
      csvContent += row + '\n';
    });
    
    // Create blob from CSV
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    
    // Create download link
    downloadFile(blob, filename);
  } catch (error) {
    console.error('Error exporting CSV:', error);
    throw new Error('Failed to export data as CSV');
  }
};

// Export data as PDF (requires external library, using a placeholder for now)
export const exportAsPDF = <T>(
  data: T, 
  filename: string = 'export.pdf', 
  title: string = 'Exported Data'
): void => {
  alert('PDF export is not yet implemented. Please use CSV or JSON export.');
  // In a real implementation, you would use a library like jsPDF to generate PDF
};

// Generate an Excel file (.xlsx)
export const exportAsExcel = <T extends Record<string, any>>(
  data: T[],
  filename: string = 'export.xlsx',
  sheetName: string = 'Sheet1',
  headers?: string[]
): void => {
  alert('Excel export is not yet implemented. Please use CSV or JSON export.');
  // In a real implementation, you would use a library like exceljs or xlsx
};

// Helper function to download a file
const downloadFile = (blob: Blob, filename: string): void => {
  // Create a URL for the blob
  const url = URL.createObjectURL(blob);
  
  // Create a link element
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  
  // Add the link to the DOM, click it, and remove it
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  
  // Release the URL object
  setTimeout(() => URL.revokeObjectURL(url), 100);
};

// Format date for filenames
export const getFormattedDate = (): string => {
  const now = new Date();
  return now.toISOString().split('T')[0]; // YYYY-MM-DD
};

// Generate a filename with date
export const generateFilename = (baseName: string, extension: string): string => {
  const dateStr = getFormattedDate();
  return `${baseName}_${dateStr}.${extension}`;
};
