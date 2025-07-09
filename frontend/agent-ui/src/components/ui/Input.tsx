// src/components/ui/Input.tsx
import type { ReactNode } from 'react'

interface InputProps {
  label?: string
  placeholder?: string
  value: string
  onChange: (value: string) => void
  type?: 'text' | 'email' | 'password' | 'number'
  disabled?: boolean
  error?: string
  icon?: ReactNode
  className?: string
}

export function Input({ 
  label, 
  placeholder, 
  value, 
  onChange, 
  type = 'text', 
  disabled = false, 
  error, 
  icon, 
  className = '' 
}: InputProps) {
  return (
    <div className={className}>
      {label && (
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          {label}
        </label>
      )}
      <div className="relative">
        {icon && (
          <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400">
            {icon}
          </div>
        )}
        <input
          type={type}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          disabled={disabled}
          className={`w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:shadow-focus outline-none transition-all ${
            icon ? 'pl-10' : ''
          } ${error ? 'border-red-500 focus:shadow-red' : ''} ${
            disabled ? 'bg-gray-50 dark:bg-gray-800 cursor-not-allowed' : ''
          }`}
          style={{
            boxShadow: error ? 'none' : undefined
          }}
          onFocus={(e) => {
            if (!error) {
              e.target.style.boxShadow = '0 0 0 2px #3b82f6'
            } else {
              e.target.style.boxShadow = '0 0 0 2px #ef4444'
            }
          }}
          onBlur={(e) => {
            e.target.style.boxShadow = 'none'
          }}
        />
      </div>
      {error && (
        <p className="mt-1 text-sm text-red-600 dark:text-red-400">{error}</p>
      )}
    </div>
  )
}
