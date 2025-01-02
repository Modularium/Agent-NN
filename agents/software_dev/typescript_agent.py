from typing import Dict, Any, Optional, List
import json
from .base_dev_agent import BaseDevAgent
from langchain.schema import Document
import subprocess
import tempfile
import os
import re

class TypeScriptDevAgent(BaseDevAgent):
    """Specialized agent for TypeScript development."""
    
    DEFAULT_STYLE_GUIDE = {
        "max_line_length": 100,
        "indent_size": 2,
        "quote_style": "single",
        "semicolons": True,
        "trailing_comma": "es5",
        "arrow_parens": "always",
        "interface_over_type": True,
        "explicit_return_type": True,
        "no_any": True
    }
    
    def __init__(self,
                 name: str = "typescript_dev",
                 domain_docs: Optional[List[Document]] = None,
                 style_guide: Optional[Dict[str, Any]] = None):
        """Initialize TypeScript development agent.
        
        Args:
            name: Agent name
            domain_docs: Optional domain knowledge documents
            style_guide: Optional coding style guide
        """
        super().__init__(
            name=name,
            language="typescript",
            domain_docs=domain_docs,
            style_guide=style_guide or self.DEFAULT_STYLE_GUIDE
        )
        
        # Initialize TypeScript tools
        self._init_ts_tools()
        
    def _init_ts_tools(self):
        """Initialize TypeScript development tools."""
        try:
            # Create temporary project
            self.project_dir = tempfile.mkdtemp()
            
            # Initialize package.json
            package_json = {
                "name": "ts-dev-agent",
                "version": "1.0.0",
                "private": true,
                "devDependencies": {
                    "typescript": "^4.9.0",
                    "prettier": "^2.8.0",
                    "eslint": "^8.0.0",
                    "@typescript-eslint/parser": "^5.0.0",
                    "@typescript-eslint/eslint-plugin": "^5.0.0"
                }
            }
            
            with open(os.path.join(self.project_dir, "package.json"), "w") as f:
                json.dump(package_json, f, indent=2)
                
            # Initialize tsconfig.json
            tsconfig = {
                "compilerOptions": {
                    "target": "es2020",
                    "module": "commonjs",
                    "strict": True,
                    "esModuleInterop": True,
                    "skipLibCheck": True,
                    "forceConsistentCasingInFileNames": True
                }
            }
            
            with open(os.path.join(self.project_dir, "tsconfig.json"), "w") as f:
                json.dump(tsconfig, f, indent=2)
                
            # Initialize ESLint config
            eslintrc = {
                "parser": "@typescript-eslint/parser",
                "plugins": ["@typescript-eslint"],
                "extends": [
                    "eslint:recommended",
                    "plugin:@typescript-eslint/recommended"
                ],
                "rules": {
                    "max-len": ["error", {"code": self.style_guide["max_line_length"]}],
                    "indent": ["error", self.style_guide["indent_size"]],
                    "quotes": ["error", self.style_guide["quote_style"]],
                    "semi": ["error", "always" if self.style_guide["semicolons"] else "never"],
                    "@typescript-eslint/explicit-function-return-type": self.style_guide["explicit_return_type"],
                    "@typescript-eslint/no-explicit-any": self.style_guide["no_any"]
                }
            }
            
            with open(os.path.join(self.project_dir, ".eslintrc.json"), "w") as f:
                json.dump(eslintrc, f, indent=2)
                
            # Install dependencies
            subprocess.run(
                ["npm", "install"],
                cwd=self.project_dir,
                check=True,
                capture_output=True
            )
            
        except Exception as e:
            self.log_error(e, {"operation": "init_ts_tools"})
            
    async def format_code(self, code: str) -> Dict[str, Any]:
        """Format TypeScript code using Prettier.
        
        Args:
            code: Code to format
            
        Returns:
            Dict[str, Any]: Formatted code and changes
        """
        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(
                suffix=".ts",
                mode="w",
                dir=self.project_dir,
                delete=False
            ) as f:
                f.write(code)
                temp_file = f.name
                
            # Run Prettier
            result = subprocess.run(
                [
                    "npx", "prettier",
                    "--write",
                    "--parser", "typescript",
                    f"--print-width={self.style_guide['max_line_length']}",
                    f"--tab-width={self.style_guide['indent_size']}",
                    f"--quote-props={self.style_guide['quote_style']}",
                    f"--trailing-comma={self.style_guide['trailing_comma']}",
                    f"--arrow-parens={self.style_guide['arrow_parens']}",
                    temp_file
                ],
                cwd=self.project_dir,
                check=True,
                capture_output=True
            )
            
            # Read formatted code
            with open(temp_file, "r") as f:
                formatted_code = f.read()
                
            # Clean up
            os.unlink(temp_file)
            
            return {
                "original_code": code,
                "formatted_code": formatted_code,
                "changes": self._diff_code(code, formatted_code)
            }
            
        except Exception as e:
            self.log_error(e, {
                "code_length": len(code),
                "operation": "format_code"
            })
            return {
                "original_code": code,
                "formatted_code": code,
                "error": str(e)
            }
            
    async def lint_code(self, code: str) -> Dict[str, Any]:
        """Lint TypeScript code using ESLint.
        
        Args:
            code: Code to lint
            
        Returns:
            Dict[str, Any]: Linting results
        """
        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(
                suffix=".ts",
                mode="w",
                dir=self.project_dir,
                delete=False
            ) as f:
                f.write(code)
                temp_file = f.name
                
            # Run ESLint
            result = subprocess.run(
                [
                    "npx", "eslint",
                    "--format", "json",
                    temp_file
                ],
                cwd=self.project_dir,
                capture_output=True,
                text=True
            )
            
            # Parse results
            if result.stdout:
                issues = json.loads(result.stdout)
            else:
                issues = []
                
            # Clean up
            os.unlink(temp_file)
            
            return {
                "issues": issues,
                "stats": {
                    "error_count": sum(1 for issue in issues if issue["severity"] == 2),
                    "warning_count": sum(1 for issue in issues if issue["severity"] == 1)
                }
            }
            
        except Exception as e:
            self.log_error(e, {
                "code_length": len(code),
                "operation": "lint_code"
            })
            return {
                "error": str(e)
            }
            
    def _analyze_code(self, code: str) -> Dict[str, Any]:
        """Enhanced code analysis for TypeScript.
        
        Args:
            code: Code to analyze
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        analysis = {
            "imports": self._analyze_imports(code),
            "exports": self._analyze_exports(code),
            "types": self._analyze_types(code),
            "functions": self._analyze_functions(code)
        }
        
        # Add basic metrics
        analysis.update(super()._analyze_code(code))
        
        return analysis
        
    def _analyze_imports(self, code: str) -> Dict[str, List[str]]:
        """Analyze TypeScript imports.
        
        Args:
            code: Code to analyze
            
        Returns:
            Dict[str, List[str]]: Import analysis
        """
        imports = {
            "named": [],
            "default": [],
            "namespace": [],
            "type": []
        }
        
        # Match different import patterns
        named_imports = re.findall(
            r"import\s*{\s*([^}]+)\s*}\s*from\s*['\"]([^'\"]+)['\"]",
            code
        )
        default_imports = re.findall(
            r"import\s+(\w+)\s+from\s*['\"]([^'\"]+)['\"]",
            code
        )
        namespace_imports = re.findall(
            r"import\s*\*\s*as\s+(\w+)\s+from\s*['\"]([^'\"]+)['\"]",
            code
        )
        type_imports = re.findall(
            r"import\s+type\s*{\s*([^}]+)\s*}\s*from\s*['\"]([^'\"]+)['\"]",
            code
        )
        
        # Process imports
        for names, source in named_imports:
            imports["named"].extend([
                {"name": n.strip(), "source": source}
                for n in names.split(",")
            ])
            
        for name, source in default_imports:
            imports["default"].append({
                "name": name,
                "source": source
            })
            
        for name, source in namespace_imports:
            imports["namespace"].append({
                "name": name,
                "source": source
            })
            
        for names, source in type_imports:
            imports["type"].extend([
                {"name": n.strip(), "source": source}
                for n in names.split(",")
            ])
            
        return imports
        
    def _analyze_exports(self, code: str) -> Dict[str, List[str]]:
        """Analyze TypeScript exports.
        
        Args:
            code: Code to analyze
            
        Returns:
            Dict[str, List[str]]: Export analysis
        """
        exports = {
            "named": [],
            "default": [],
            "type": []
        }
        
        # Match different export patterns
        named_exports = re.findall(
            r"export\s*{\s*([^}]+)\s*}",
            code
        )
        default_exports = re.findall(
            r"export\s+default\s+(?:class|function|const|let|var)?\s*(\w+)",
            code
        )
        type_exports = re.findall(
            r"export\s+type\s+(\w+)",
            code
        )
        
        # Process exports
        for names in named_exports:
            exports["named"].extend([n.strip() for n in names.split(",")])
            
        exports["default"].extend(default_exports)
        exports["type"].extend(type_exports)
        
        return exports
        
    def _analyze_types(self, code: str) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze TypeScript types and interfaces.
        
        Args:
            code: Code to analyze
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Type analysis
        """
        types = {
            "interfaces": [],
            "types": [],
            "enums": []
        }
        
        # Match interfaces
        interface_matches = re.finditer(
            r"interface\s+(\w+)(?:\s+extends\s+([^{]+))?\s*{([^}]+)}",
            code
        )
        for match in interface_matches:
            types["interfaces"].append({
                "name": match.group(1),
                "extends": match.group(2).strip() if match.group(2) else None,
                "properties": self._parse_properties(match.group(3))
            })
            
        # Match type aliases
        type_matches = re.finditer(
            r"type\s+(\w+)(?:\s*<[^>]+>)?\s*=\s*([^;]+)",
            code
        )
        for match in type_matches:
            types["types"].append({
                "name": match.group(1),
                "definition": match.group(2).strip()
            })
            
        # Match enums
        enum_matches = re.finditer(
            r"enum\s+(\w+)\s*{([^}]+)}",
            code
        )
        for match in enum_matches:
            types["enums"].append({
                "name": match.group(1),
                "values": [
                    v.strip()
                    for v in match.group(2).split(",")
                    if v.strip()
                ]
            })
            
        return types
        
    def _analyze_functions(self, code: str) -> List[Dict[str, Any]]:
        """Analyze TypeScript functions.
        
        Args:
            code: Code to analyze
            
        Returns:
            List[Dict[str, Any]]: Function analysis
        """
        functions = []
        
        # Match function declarations
        function_matches = re.finditer(
            r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*"
            r"<([^>]+)>?\s*\(([^)]*)\)\s*:\s*([^{]+)",
            code
        )
        
        for match in function_matches:
            functions.append({
                "name": match.group(1),
                "generic_params": match.group(2) if match.group(2) else None,
                "parameters": self._parse_parameters(match.group(3)),
                "return_type": match.group(4).strip()
            })
            
        return functions
        
    def _parse_properties(self, properties_str: str) -> List[Dict[str, str]]:
        """Parse interface properties.
        
        Args:
            properties_str: Property definitions string
            
        Returns:
            List[Dict[str, str]]: Parsed properties
        """
        properties = []
        
        for line in properties_str.split(";"):
            line = line.strip()
            if not line:
                continue
                
            # Match property pattern
            match = re.match(
                r"(?:readonly\s+)?(\w+)\??\s*:\s*([^;]+)",
                line
            )
            if match:
                properties.append({
                    "name": match.group(1),
                    "type": match.group(2).strip()
                })
                
        return properties
        
    def _parse_parameters(self, params_str: str) -> List[Dict[str, str]]:
        """Parse function parameters.
        
        Args:
            params_str: Parameter definitions string
            
        Returns:
            List[Dict[str, str]]: Parsed parameters
        """
        parameters = []
        
        if not params_str.strip():
            return parameters
            
        for param in params_str.split(","):
            param = param.strip()
            if not param:
                continue
                
            # Match parameter pattern
            match = re.match(
                r"(?:readonly\s+)?(\w+)\??\s*:\s*([^=]+)(?:\s*=\s*(.+))?",
                param
            )
            if match:
                parameters.append({
                    "name": match.group(1),
                    "type": match.group(2).strip(),
                    "default": match.group(3).strip() if match.group(3) else None
                })
                
        return parameters