// Card Component
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
    <div className={`bg-white rounded-xl border border-gray-200 ${paddings[padding]} ${
      hover ? 'hover:shadow-lg transition-shadow duration-200' : 'shadow-sm'
    } ${className}`}>
      {(title || subtitle) && (
        <div className="mb-4">
          {title && <h3 className="text-lg font-semibold text-gray-900">{title}</h3>}
          {subtitle && <p className="text-gray-600">{subtitle}</p>}
        </div>
      )}
      {children}
    </div>
  )
}
