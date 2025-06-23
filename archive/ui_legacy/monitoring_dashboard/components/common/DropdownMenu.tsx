// monitoring/dashboard/components/common/DropdownMenu.tsx
import React, { useState, useRef, useEffect } from 'react';
import { ChevronDown, Check, ChevronRight } from 'lucide-react';

export interface DropdownMenuItem {
  id: string;
  label: React.ReactNode;
  icon?: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  href?: string;
  divider?: boolean;
  subItems?: DropdownMenuItem[];
  selected?: boolean;
  variant?: 'default' | 'danger' | 'warning' | 'success';
  description?: string;
}

export interface DropdownMenuProps {
  trigger: React.ReactNode;
  items: DropdownMenuItem[];
  className?: string;
  menuClassName?: string;
  triggerClassName?: string;
  itemClassName?: string;
  align?: 'left' | 'right';
  width?: 'auto' | 'sm' | 'md' | 'lg' | 'xl';
  isOpen?: boolean;
  setIsOpen?: (isOpen: boolean) => void;
  closeOnClick?: boolean;
  showIcons?: boolean;
  maxHeight?: number | string;
}

const DropdownMenu: React.FC<DropdownMenuProps> = ({
  trigger,
  items,
  className = '',
  menuClassName = '',
  triggerClassName = '',
  itemClassName = '',
  align = 'left',
  width = 'auto',
  isOpen: controlledIsOpen,
  setIsOpen: setControlledIsOpen,
  closeOnClick = true,
  showIcons = true,
  maxHeight,
}) => {
  // Support both controlled and uncontrolled mode
  const isControlled = controlledIsOpen !== undefined && setControlledIsOpen !== undefined;
  const [uncontrolledIsOpen, setUncontrolledIsOpen] = useState(false);
  
  // Use either controlled or uncontrolled state
  const isOpen = isControlled ? controlledIsOpen : uncontrolledIsOpen;
  const setIsOpen = isControlled 
    ? setControlledIsOpen 
    : setUncontrolledIsOpen;
    
  const [activeSubMenu, setActiveSubMenu] = useState<string | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  
  // Handle click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current && 
        !dropdownRef.current.contains(event.target as Node) &&
        isOpen
      ) {
        setIsOpen(false);
        setActiveSubMenu(null);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen, setIsOpen]);
  
  // Handle escape key to close dropdown
  useEffect(() => {
    const handleEscapeKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && isOpen) {
        setIsOpen(false);
        setActiveSubMenu(null);
      }
    };
    
    document.addEventListener('keydown', handleEscapeKey);
    return () => {
      document.removeEventListener('keydown', handleEscapeKey);
    };
  }, [isOpen, setIsOpen]);
  
  // Toggle dropdown
  const toggleDropdown = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsOpen(!isOpen);
    if (!isOpen) {
      setActiveSubMenu(null);
    }
  };
  
  // Handle item click
  const handleItemClick = (item: DropdownMenuItem, e: React.MouseEvent) => {
    if (item.disabled) {
      e.preventDefault();
      return;
    }
    
    // For items with submenus, toggle the submenu
    if (item.subItems && item.subItems.length > 0) {
      e.preventDefault();
      e.stopPropagation();
      setActiveSubMenu(activeSubMenu === item.id ? null : item.id);
      return;
    }
    
    // For normal items, execute onClick and close dropdown if needed
    if (item.onClick) {
      item.onClick();
    }
    
    if (closeOnClick) {
      setIsOpen(false);
      setActiveSubMenu(null);
    }
  };
  
  // Width classes
  const widthClasses = {
    auto: 'w-auto',
    sm: 'w-40',
    md: 'w-48',
    lg: 'w-56',
    xl: 'w-64',
  };
  
  // Variant classes
  const getItemVariantClasses = (variant: DropdownMenuItem['variant'] = 'default') => {
    switch (variant) {
      case 'danger':
        return 'text-red-700 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20';
      case 'warning':
        return 'text-yellow-700 dark:text-yellow-400 hover:bg-yellow-50 dark:hover:bg-yellow-900/20';
      case 'success':
        return 'text-green-700 dark:text-green-400 hover:bg-green-50 dark:hover:bg-green-900/20';
      default:
        return 'text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700';
    }
  };
  
  // Render a menu item
  const renderMenuItem = (item: DropdownMenuItem) => {
    // For dividers
    if (item.divider) {
      return <div key={item.id} className="my-1 border-t border-gray-200 dark:border-gray-700" />;
    }
    
    // Common classes for all item types
    const commonClasses = `
      ${itemClassName}
      ${getItemVariantClasses(item.variant)}
      ${item.disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
      text-sm flex items-center px-4 py-2 
    `;
    
    // Element used depends on if there's an href
    const Element = item.href ? 'a' : 'div';
    
    // Props for the element
    const elementProps: any = {
      className: commonClasses,
      onClick: (e: React.MouseEvent) => handleItemClick(item, e),
    };
    
    // Add href if it exists
    if (item.href && !item.disabled) {
      elementProps.href = item.href;
    }
    
    return (
      <Element
        key={item.id}
        {...elementProps}
      >
        <div className="flex-1 flex items-center min-w-0">
          {/* Icon */}
          {showIcons && (
            <div className="flex-shrink-0 w-5 h-5 mr-3 flex items-center justify-center">
              {item.icon}
              {item.selected && !item.icon && (
                <Check size={16} className="text-indigo-600 dark:text-indigo-400" />
              )}
            </div>
          )}
          
          {/* Label and description */}
          <div className="min-w-0 flex-1">
            <div className="truncate">{item.label}</div>
            {item.description && (
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5 truncate">
                {item.description}
              </p>
            )}
          </div>
          
          {/* Submenu indicator */}
          {item.subItems && item.subItems.length > 0 && (
            <ChevronRight size={16} className="ml-2 text-gray-400" />
          )}
        </div>
      </Element>
    );
  };
  
  // Render submenu
  const renderSubMenu = (parentItem: DropdownMenuItem) => {
    if (!parentItem.subItems || activeSubMenu !== parentItem.id) {
      return null;
    }
    
    return (
      <div 
        className={`
          absolute top-0 left-full ml-0.5 ${widthClasses[width]}
          bg-white dark:bg-gray-800 rounded-md shadow-lg ring-1 ring-black ring-opacity-5 z-20
          ${menuClassName}
        `}
        style={{ maxHeight: maxHeight ? maxHeight : 'auto', overflowY: maxHeight ? 'auto' : 'visible' }}
      >
        {parentItem.subItems.map(subItem => renderMenuItem(subItem))}
      </div>
    );
  };
  
  return (
    <div className={`relative inline-block text-left ${className}`} ref={dropdownRef}>
      {/* Trigger */}
      <div 
        className={`inline-flex w-full justify-center items-center cursor-pointer ${triggerClassName}`}
        onClick={toggleDropdown}
      >
        {typeof trigger === 'string' ? (
          <button 
            type="button" 
            className="inline-flex w-full justify-center items-center px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none"
          >
            {trigger}
            <ChevronDown size={16} className="ml-2 -mr-1" />
          </button>
        ) : (
          trigger
        )}
      </div>
      
      {/* Dropdown Menu */}
      {isOpen && (
        <div 
          className={`
            absolute z-10 mt-1 ${widthClasses[width]}
            bg-white dark:bg-gray-800 rounded-md shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none
            ${align === 'left' ? 'origin-top-left left-0' : 'origin-top-right right-0'}
            ${menuClassName}
          `}
          style={{ maxHeight: maxHeight ? maxHeight : 'auto', overflowY: maxHeight ? 'auto' : 'visible' }}
        >
          <div className="py-1">
            {items.map(item => (
              <div key={item.id} className="relative">
                {renderMenuItem(item)}
                {renderSubMenu(item)}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default DropdownMenu;
