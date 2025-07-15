#!/usr/bin/env bash
set -e

errors=0
while IFS= read -r file; do
    case "$file" in
        *.sh)
            bash -n "$file" || { echo "Syntaxfehler in $file" >&2; errors=$((errors+1)); }
            ;;
        *.py)
            python -m py_compile "$file" 2>/dev/null || { echo "Syntaxfehler in $file" >&2; errors=$((errors+1)); }
            ;;
    esac
done < <(git ls-files 'scripts/**/*.sh' 'scripts/**/*.py')

if [[ $errors -gt 0 ]]; then
    echo "$errors Fehler gefunden"
    exit 1
else
    echo "Alle Skripte OK"
fi
