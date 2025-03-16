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
  const [un
