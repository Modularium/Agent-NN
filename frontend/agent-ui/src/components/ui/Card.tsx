// src/components/ui/Card.tsx
import type { ReactNode } from 'react'

interface CardProps {
  children: ReactNode
  title?: string
  subtitle?: string
  padding?: 'sm' | 'md' | 'lg'
  hover?: boolean
  className?: string
}

export function Card({ 
  children, 
  title, 
  subtitle, 
  padding = 'md', 
  hover = false, 
  className = '' 
}: CardProps) {
  const paddings = {
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8'
  }

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 ${paddings[padding]} ${
      hover ? 'hover:shadow-lg transition-shadow duration-200' : 'shadow-sm'
    } ${className}`}>
      {(title || subtitle) && (
        <div className="mb-4">
          {title && <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{title}</h3>}
          {subtitle && <p className="text-gray-600 dark:text-gray-400">{subtitle}</p>}
        </div>
      )}
      {children}
    </div>
  )
}
