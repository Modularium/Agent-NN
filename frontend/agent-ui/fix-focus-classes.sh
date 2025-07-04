#!/bin/bash

# Fix focus:ring classes for Tailwind v4 compatibility
# Run this script from the frontend/agent-ui directory

echo "🔧 Fixing focus:ring classes for Tailwind v4 compatibility..."

# Function to replace in files
replace_in_files() {
    find src -name "*.tsx" -o -name "*.ts" | xargs grep -l "$1" | while read file; do
        echo "Fixing: $file"
        sed -i "s/$1/$2/g" "$file"
    done
}

# Replace common focus:ring patterns
echo "Replacing focus:ring-2 focus:ring-blue-500 focus:ring-offset-2..."
replace_in_files "focus:ring-2 focus:ring-blue-500 focus:ring-offset-2" "focus:outline-none focus:shadow-focus"

echo "Replacing focus:ring-2 focus:ring-blue-500..."
replace_in_files "focus:ring-2 focus:ring-blue-500" "focus:outline-none focus:shadow-focus"

echo "Replacing focus:ring-2 focus:ring-red-500..."
replace_in_files "focus:ring-2 focus:ring-red-500" "focus:outline-none focus:shadow-red"

echo "Replacing focus:ring-2 focus:ring-green-500..."
replace_in_files "focus:ring-2 focus:ring-green-500" "focus:outline-none focus:shadow-green"

echo "Replacing focus:ring-2 focus:ring-gray-500..."
replace_in_files "focus:ring-2 focus:ring-gray-500" "focus:outline-none focus:shadow-focus"

echo "Replacing remaining focus:ring-offset-2 patterns..."
replace_in_files "focus:ring-offset-2" ""

# Handle any remaining focus:ring classes
echo "Replacing any remaining focus:ring classes..."
find src -name "*.tsx" -o -name "*.ts" | xargs grep -l "focus:ring-" | while read file; do
    echo "Manual review needed for remaining focus:ring classes in: $file"
    grep -n "focus:ring-" "$file" || true
done

echo "✅ Focus class replacement complete!"
echo ""
echo "Next steps:"
echo "1. Review any files listed above for manual fixes"
echo "2. Replace your vite.config.ts, tailwind.config.js, and src/index.css with the fixed versions"
echo "3. Replace src/components/ui/Button.tsx and src/components/ui/Input.tsx with fixed versions"
echo "4. Run 'npm run dev' to test"
