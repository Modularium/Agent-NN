from typing import List, Dict, Any, Optional, Set
import re
import ast
import json

class CodeSafetyValidator:
    """Validator to ensure code safety and prevent execution."""
    
    # Dangerous Python builtins and modules
    DANGEROUS_BUILTINS = {
        'eval', 'exec', 'compile', '__import__',
        'globals', 'locals', 'vars', 'dir',
        'getattr', 'setattr', 'delattr',
        'open', 'input', 'print'  # IO operations
    }
    
    DANGEROUS_MODULES = {
        'os', 'sys', 'subprocess', 'multiprocessing',
        'socket', 'requests', 'urllib', 'ftplib',
        'telnetlib', 'smtplib', 'pickle', 'shelve',
        'marshal', 'base64', 'codecs', 'crypt',
        'pwd', 'grp', 'crypt', 'spwd', 'crypt'
    }
    
    # Dangerous JavaScript/TypeScript patterns
    DANGEROUS_JS_PATTERNS = [
        r'eval\s*\(',
        r'Function\s*\(',
        r'setTimeout\s*\(',
        r'setInterval\s*\(',
        r'new\s+Function',
        r'document\.',
        r'window\.',
        r'localStorage\.',
        r'sessionStorage\.',
        r'indexedDB\.',
        r'fetch\s*\(',
        r'XMLHttpRequest',
        r'WebSocket',
        r'Worker\s*\(',
        r'SharedWorker\s*\(',
        r'process\.',
        r'require\s*\(',
        r'import\s*\(',
        r'fs\.',
        r'child_process',
        r'cluster\.',
        r'crypto\.',
        r'http\.',
        r'https\.',
        r'net\.',
        r'dgram\.',
        r'dns\.'
    ]
    
    def __init__(self):
        """Initialize code safety validator."""
        # Compile regex patterns
        self.js_patterns = [re.compile(p) for p in self.DANGEROUS_JS_PATTERNS]
        
    def validate_python(self, code: str) -> List[Dict[str, Any]]:
        """Validate Python code for safety.
        
        Args:
            code: Python code to validate
            
        Returns:
            List[Dict[str, Any]]: List of safety violations
        """
        violations = []
        
        try:
            # Parse code into AST
            tree = ast.parse(code)
            
            # Check for dangerous patterns
            for node in ast.walk(tree):
                # Check function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.DANGEROUS_BUILTINS:
                            violations.append({
                                "type": "dangerous_builtin",
                                "name": node.func.id,
                                "line": node.lineno,
                                "col": node.col_offset,
                                "severity": "high"
                            })
                            
                # Check imports
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for name in node.names:
                        module = name.name.split('.')[0]
                        if module in self.DANGEROUS_MODULES:
                            violations.append({
                                "type": "dangerous_import",
                                "name": module,
                                "line": node.lineno,
                                "col": node.col_offset,
                                "severity": "high"
                            })
                            
                # Check exec/eval strings
                elif isinstance(node, ast.Str):
                    if any(keyword in node.s.lower() 
                          for keyword in ['exec', 'eval', 'import']):
                        violations.append({
                            "type": "suspicious_string",
                            "content": node.s[:50],
                            "line": node.lineno,
                            "col": node.col_offset,
                            "severity": "medium"
                        })
                        
        except SyntaxError as e:
            violations.append({
                "type": "syntax_error",
                "message": str(e),
                "line": e.lineno,
                "col": e.offset,
                "severity": "high"
            })
            
        except Exception as e:
            violations.append({
                "type": "validation_error",
                "message": str(e),
                "severity": "high"
            })
            
        return violations
        
    def validate_javascript(self, code: str) -> List[Dict[str, Any]]:
        """Validate JavaScript/TypeScript code for safety.
        
        Args:
            code: JavaScript/TypeScript code to validate
            
        Returns:
            List[Dict[str, Any]]: List of safety violations
        """
        violations = []
        
        # Split code into lines for better error reporting
        lines = code.split('\n')
        
        # Check each pattern
        for pattern in self.js_patterns:
            for i, line in enumerate(lines, 1):
                match = pattern.search(line)
                if match:
                    violations.append({
                        "type": "dangerous_pattern",
                        "pattern": pattern.pattern,
                        "line": i,
                        "col": match.start(),
                        "content": line.strip(),
                        "severity": "high"
                    })
                    
        # Check for inline scripts
        if re.search(r'<script.*?>', code, re.DOTALL):
            violations.append({
                "type": "inline_script",
                "message": "Inline scripts detected",
                "severity": "medium"
            })
            
        # Check for dangerous URLs
        urls = re.finditer(
            r'(?:href|src|url)\s*=\s*["\']([^"\']+)["\']',
            code
        )
        for match in urls:
            url = match.group(1)
            if not url.startswith(('http://', 'https://', '/')):
                violations.append({
                    "type": "suspicious_url",
                    "url": url,
                    "line": len(code[:match.start()].split('\n')),
                    "severity": "medium"
                })
                
        return violations
        
    def validate_html(self, code: str) -> List[Dict[str, Any]]:
        """Validate HTML code for safety.
        
        Args:
            code: HTML code to validate
            
        Returns:
            List[Dict[str, Any]]: List of safety violations
        """
        violations = []
        
        # Check for inline scripts
        script_tags = re.finditer(
            r'<script\b[^>]*>(.*?)</script>',
            code,
            re.DOTALL
        )
        for match in script_tags:
            violations.append({
                "type": "inline_script",
                "content": match.group(1)[:50],
                "line": len(code[:match.start()].split('\n')),
                "severity": "high"
            })
            
        # Check for event handlers
        event_handlers = re.finditer(
            r'\bon\w+\s*=\s*["\'][^"\']*["\']',
            code
        )
        for match in event_handlers:
            violations.append({
                "type": "event_handler",
                "content": match.group(0),
                "line": len(code[:match.start()].split('\n')),
                "severity": "high"
            })
            
        # Check for dangerous URLs
        urls = re.finditer(
            r'(?:href|src|action)\s*=\s*["\']([^"\']+)["\']',
            code
        )
        for match in urls:
            url = match.group(1)
            if not url.startswith(('http://', 'https://', '/')):
                violations.append({
                    "type": "suspicious_url",
                    "url": url,
                    "line": len(code[:match.start()].split('\n')),
                    "severity": "medium"
                })
                
        return violations
        
    def validate_css(self, code: str) -> List[Dict[str, Any]]:
        """Validate CSS code for safety.
        
        Args:
            code: CSS code to validate
            
        Returns:
            List[Dict[str, Any]]: List of safety violations
        """
        violations = []
        
        # Check for imports
        imports = re.finditer(r'@import\s+["\']([^"\']+)["\']', code)
        for match in imports:
            url = match.group(1)
            if not url.startswith(('http://', 'https://', '/')):
                violations.append({
                    "type": "suspicious_import",
                    "url": url,
                    "line": len(code[:match.start()].split('\n')),
                    "severity": "medium"
                })
                
        # Check for URLs in properties
        urls = re.finditer(r'url\s*\(\s*["\']?([^"\']+)["\']?\s*\)', code)
        for match in urls:
            url = match.group(1)
            if not url.startswith(('http://', 'https://', '/', 'data:')):
                violations.append({
                    "type": "suspicious_url",
                    "url": url,
                    "line": len(code[:match.start()].split('\n')),
                    "severity": "medium"
                })
                
        # Check for expression functions
        expressions = re.finditer(r'expression\s*\(', code)
        for match in expressions:
            violations.append({
                "type": "dangerous_expression",
                "line": len(code[:match.start()].split('\n')),
                "severity": "high"
            })
            
        return violations
        
    def is_code_safe(self,
                    code: str,
                    language: str,
                    strict: bool = True) -> bool:
        """Check if code is safe to process.
        
        Args:
            code: Code to validate
            language: Programming language
            strict: Whether to use strict validation
            
        Returns:
            bool: Whether code is safe
        """
        # Get appropriate validator
        validators = {
            "python": self.validate_python,
            "javascript": self.validate_javascript,
            "typescript": self.validate_javascript,
            "html": self.validate_html,
            "css": self.validate_css
        }
        
        validator = validators.get(language.lower())
        if not validator:
            return False
            
        # Get violations
        violations = validator(code)
        
        if strict:
            # Any violation makes code unsafe
            return len(violations) == 0
        else:
            # Only high severity violations make code unsafe
            return not any(v["severity"] == "high" for v in violations)
            
    def get_safe_subset(self,
                       code: str,
                       language: str) -> Optional[str]:
        """Get safe subset of code by removing dangerous parts.
        
        Args:
            code: Code to process
            language: Programming language
            
        Returns:
            Optional[str]: Safe subset of code or None if not possible
        """
        if language.lower() == "python":
            try:
                # Parse code
                tree = ast.parse(code)
                
                # Remove dangerous nodes
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            if node.func.id in self.DANGEROUS_BUILTINS:
                                # Replace with pass
                                new_node = ast.Pass()
                                ast.copy_location(new_node, node)
                                node.func = new_node
                                
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        for name in node.names:
                            if name.name.split('.')[0] in self.DANGEROUS_MODULES:
                                # Remove import
                                node.names.remove(name)
                                
                # Generate code
                return ast.unparse(tree)
                
            except Exception:
                return None
                
        elif language.lower() in ["javascript", "typescript"]:
            # Remove dangerous patterns
            safe_code = code
            for pattern in self.js_patterns:
                safe_code = pattern.sub('', safe_code)
            return safe_code
            
        elif language.lower() == "html":
            # Remove script tags and event handlers
            safe_code = re.sub(
                r'<script\b[^>]*>.*?</script>',
                '',
                code,
                flags=re.DOTALL
            )
            safe_code = re.sub(
                r'\bon\w+\s*=\s*["\'][^"\']*["\']',
                '',
                safe_code
            )
            return safe_code
            
        elif language.lower() == "css":
            # Remove expressions
            safe_code = re.sub(
                r'expression\s*\([^)]*\)',
                '',
                code
            )
            return safe_code
            
        return None