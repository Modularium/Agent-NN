from typing import Dict, Any, Optional, List
import ast
import astroid
from pylint.lint import Run
from pylint.reporters import JSONReporter
import black
import isort
from io import StringIO
from .base_dev_agent import BaseDevAgent
from langchain.schema import Document

class PythonDevAgent(BaseDevAgent):
    """Specialized agent for Python development."""
    
    DEFAULT_STYLE_GUIDE = {
        "max_line_length": 88,  # Black's default
        "indent_size": 4,
        "quote_style": "double",
        "docstring_style": "google",
        "import_order": [
            "future",
            "standard_library",
            "third_party",
            "first_party",
            "local_folder"
        ]
    }
    
    def __init__(self,
                 name: str = "python_dev",
                 domain_docs: Optional[List[Document]] = None,
                 style_guide: Optional[Dict[str, Any]] = None):
        """Initialize Python development agent.
        
        Args:
            name: Agent name
            domain_docs: Optional domain knowledge documents
            style_guide: Optional coding style guide
        """
        super().__init__(
            name=name,
            language="python",
            domain_docs=domain_docs,
            style_guide=style_guide or self.DEFAULT_STYLE_GUIDE
        )
        
    async def format_code(self, code: str) -> Dict[str, Any]:
        """Format Python code using Black and isort.
        
        Args:
            code: Code to format
            
        Returns:
            Dict[str, Any]: Formatted code and changes
        """
        try:
            # Format with Black
            black_mode = black.FileMode(
                line_length=self.style_guide["max_line_length"]
            )
            black_formatted = black.format_str(code, mode=black_mode)
            
            # Sort imports
            isort_config = {
                "line_length": self.style_guide["max_line_length"],
                "multi_line_output": 3,  # Vertical Hanging Indent
                "include_trailing_comma": True,
                "force_grid_wrap": 0,
                "use_parentheses": True,
                "ensure_newline_before_comments": True,
                "sections": self.style_guide["import_order"]
            }
            final_code = isort.code(black_formatted, **isort_config)
            
            return {
                "original_code": code,
                "formatted_code": final_code,
                "changes": self._diff_code(code, final_code)
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
        """Lint Python code using Pylint.
        
        Args:
            code: Code to lint
            
        Returns:
            Dict[str, Any]: Linting results
        """
        try:
            # Create temporary file-like object
            code_file = StringIO(code)
            code_file.name = "temp.py"  # Pylint needs a name
            
            # Set up reporter
            reporter = JSONReporter()
            
            # Run Pylint
            Run(
                ["--output-format=json", code_file.name],
                reporter=reporter,
                exit=False
            )
            
            # Get results
            results = reporter.data
            
            return {
                "issues": results,
                "stats": {
                    "error_count": len([r for r in results if r["type"] == "error"]),
                    "warning_count": len([r for r in results if r["type"] == "warning"]),
                    "convention_count": len([r for r in results if r["type"] == "convention"])
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
        """Enhanced code analysis for Python.
        
        Args:
            code: Code to analyze
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            # Parse with astroid for more detailed analysis
            module = astroid.parse(code)
            
            analysis = {
                "num_functions": len(list(module.nodes_of_class(astroid.FunctionDef))),
                "num_classes": len(list(module.nodes_of_class(astroid.ClassDef))),
                "num_methods": len([
                    node for node in module.nodes_of_class(astroid.FunctionDef)
                    if node.is_method()
                ]),
                "imports": self._analyze_imports(module),
                "complexity": self._calculate_complexity(ast.parse(code)),
                "docstring_coverage": self._analyze_docstrings(module)
            }
            
            return analysis
            
        except Exception as e:
            self.log_error(e, {
                "code_length": len(code),
                "operation": "analyze_code"
            })
            
            # Fallback to basic analysis
            return super()._analyze_code(code)
            
    def _analyze_imports(self, module: astroid.Module) -> Dict[str, List[str]]:
        """Analyze Python imports.
        
        Args:
            module: Astroid module
            
        Returns:
            Dict[str, List[str]]: Import analysis
        """
        imports = {
            "standard_library": [],
            "third_party": [],
            "local": []
        }
        
        for node in module.nodes_of_class((astroid.Import, astroid.ImportFrom)):
            for name in node.names:
                module_name = name[0].split(".")[0]
                try:
                    __import__(module_name)
                    if module_name in __builtins__.__dict__:
                        imports["standard_library"].append(name[0])
                    else:
                        imports["third_party"].append(name[0])
                except ImportError:
                    imports["local"].append(name[0])
                    
        return imports
        
    def _analyze_docstrings(self, module: astroid.Module) -> Dict[str, float]:
        """Analyze Python docstring coverage.
        
        Args:
            module: Astroid module
            
        Returns:
            Dict[str, float]: Docstring coverage stats
        """
        stats = {
            "total": 0,
            "with_docstring": 0
        }
        
        for node in module.nodes_of_class((
            astroid.FunctionDef,
            astroid.ClassDef,
            astroid.Module
        )):
            stats["total"] += 1
            if node.doc:
                stats["with_docstring"] += 1
                
        return {
            "coverage": stats["with_docstring"] / stats["total"] if stats["total"] > 0 else 1.0,
            **stats
        }