// src/components/ui/Button.tsx
import React, { forwardRef, ReactNode } from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/utils/cn'

const buttonVariants = cva(
  "inline-flex items-center justify-center font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 rounded-lg gap-2",
  {
    variants: {
      variant: {
        default: "bg-blue-600 text-white shadow-sm hover:bg-blue-700 active:bg-blue-800 shadow-blue-600/25",
        destructive: "bg-red-600 text-white shadow-sm hover:bg-red-700 active:bg-red-800 shadow-red-600/25",
        outline: "border border-gray-300 bg-white text-gray-700 shadow-sm hover:bg-gray-50 hover:border-gray-400 active:bg-gray-100",
        secondary: "bg-gray-100 text-gray-900 shadow-sm hover:bg-gray-200 active:bg-gray-300",
        ghost: "text-gray-700 hover:bg-gray-100 active:bg-gray-200",
        success: "bg-green-600 text-white shadow-sm hover:bg-green-700 active:bg-green-800 shadow-green-600/25",
        warning: "bg-yellow-600 text-white shadow-sm hover:bg-yellow-700 active:bg-yellow-800 shadow-yellow-600/25"
      },
      size: {
        default: "h-10 px-4 py-2 text-sm",
        sm: "h-8 px-3 py-1.5 text-xs",
        lg: "h-12 px-6 py-3 text-base",
        xl: "h-14 px-8 py-4 text-lg",
        icon: "h-10 w-10"
      },
      fullWidth: {
        true: "w-full"
      }
    },
    defaultVariants: {
      variant: "default",
      size: "default"
    }
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
  loading?: boolean
  leftIcon?: ReactNode
  rightIcon?: ReactNode
  children?: ReactNode
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ 
    className, 
    variant, 
    size, 
    fullWidth,
    loading = false, 
    disabled, 
    leftIcon, 
    rightIcon, 
    children, 
    ...props 
  }, ref) => {
    const isDisabled = disabled || loading

    return (
      <button
        className={cn(buttonVariants({ variant, size, fullWidth, className }))}
        ref={ref}
        disabled={isDisabled}
        aria-disabled={isDisabled}
        {...props}
      >
        {loading && (
          <svg
            className="animate-spin h-4 w-4"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            ></circle>
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            ></path>
          </svg>
        )}
        {!loading && leftIcon && leftIcon}
        {children}
        {!loading && rightIcon && rightIcon}
      </button>
    )
  }
)

Button.displayName = "Button"

export { Button, buttonVariants }
