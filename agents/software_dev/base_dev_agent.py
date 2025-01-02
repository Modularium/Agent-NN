from typing import List, Dict, Any, Optional, Union
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from agents.worker_agent import WorkerAgent
from utils.logging_util import LoggerMixin
from .safety_validator import CodeSafetyValidator
import re
import ast
import json

class BaseDevAgent(WorkerAgent, LoggerMixin):
    """Base agent for software development tasks."""
    
    def __init__(self,
                 name: str,
                 language: str,
                 domain_docs: Optional[List[Document]] = None,
                 style_guide: Optional[Dict[str, Any]] = None):
        """Initialize software development agent.
        
        Args:
            name: Agent name
            language: Programming language
            domain_docs: Optional domain knowledge documents
            style_guide: Optional coding style guide
        """
        super().__init__(name=name, domain_docs=domain_docs)
        self.language = language
        self.style_guide = style_guide or {}
        
        # Initialize code analysis tools
        self.code_analyzer = self._get_code_analyzer()
        
        # Initialize safety validator
        self.safety_validator = CodeSafetyValidator()
        
        # Load language-specific templates
        self.templates = self._load_templates()
        
    def _get_code_analyzer(self):
        """Get appropriate code analyzer for language."""
        analyzers = {
            "python": ast,
            # Add other language parsers as needed
        }
        return analyzers.get(self.language.lower())
        
    def _load_templates(self) -> Dict[str, PromptTemplate]:
        """Load language-specific prompt templates."""
        return {
            "code_generation": PromptTemplate(
                template="""Generate {language} code for the following task:
                
                Task: {task_description}
                
                Requirements:
                {requirements}
                
                Style Guide:
                {style_guide}
                
                Additional Context:
                {context}
                
                Please provide well-structured, documented code that follows the style guide.
                """,
                input_variables=[
                    "language",
                    "task_description",
                    "requirements",
                    "style_guide",
                    "context"
                ]
            ),
            "code_review": PromptTemplate(
                template="""Review the following {language} code:
                
                Code:
                {code}
                
                Style Guide:
                {style_guide}
                
                Please provide:
                1. Code quality assessment
                2. Potential issues or bugs
                3. Style guide violations
                4. Suggested improvements
                """,
                input_variables=[
                    "language",
                    "code",
                    "style_guide"
                ]
            ),
            "code_refactor": PromptTemplate(
                template="""Refactor the following {language} code:
                
                Original Code:
                {code}
                
                Refactoring Goals:
                {goals}
                
                Style Guide:
                {style_guide}
                
                Please provide:
                1. Refactored code
                2. Explanation of changes
                3. Benefits of the refactoring
                """,
                input_variables=[
                    "language",
                    "code",
                    "goals",
                    "style_guide"
                ]
            )
        }
        
    async def generate_code(self,
                          task_description: str,
                          requirements: List[str],
                          context: Optional[str] = None) -> Dict[str, Any]:
        """Generate code based on task description.
        
        Args:
            task_description: Description of coding task
            requirements: List of requirements
            context: Optional additional context
            
        Returns:
            Dict[str, Any]: Generated code and metadata
        """
        # Format requirements and style guide
        req_text = "\n".join(f"- {req}" for req in requirements)
        style_text = json.dumps(self.style_guide, indent=2)
        
        # Generate prompt
        prompt = self.templates["code_generation"].format(
            language=self.language,
            task_description=task_description,
            requirements=req_text,
            style_guide=style_text,
            context=context or "None provided"
        )
        
        # Get response from LLM
        response = await self.llm.agenerate(prompt)
        
        # Extract code from response
        code = self._extract_code(response)
        
        # Validate code safety
        violations = self.safety_validator.validate_python(code)
        if violations:
            self.log_event(
                "code_safety_violation",
                {
                    "violations": violations,
                    "code_length": len(code)
                }
            )
            # Get safe subset or return error
            safe_code = self.safety_validator.get_safe_subset(code, self.language)
            if not safe_code:
                return {
                    "error": "Generated code contains unsafe patterns",
                    "violations": violations
                }
            code = safe_code
        
        # Analyze code
        analysis = self._analyze_code(code)
        
        return {
            "code": code,
            "language": self.language,
            "analysis": analysis,
            "metadata": {
                "task": task_description,
                "requirements": requirements,
                "timestamp": self.llm.get_timestamp()
            }
        }
        
    async def review_code(self, code: str) -> Dict[str, Any]:
        """Review code for quality and issues.
        
        Args:
            code: Code to review
            
        Returns:
            Dict[str, Any]: Review results
        """
        # Generate prompt
        prompt = self.templates["code_review"].format(
            language=self.language,
            code=code,
            style_guide=json.dumps(self.style_guide, indent=2)
        )
        
        # Get response from LLM
        response = await self.llm.agenerate(prompt)
        
        # Analyze code
        analysis = self._analyze_code(code)
        
        return {
            "review": response,
            "analysis": analysis,
            "style_violations": self._check_style(code),
            "metrics": self._calculate_metrics(code)
        }
        
    async def refactor_code(self,
                          code: str,
                          goals: List[str]) -> Dict[str, Any]:
        """Refactor code according to specified goals.
        
        Args:
            code: Code to refactor
            goals: Refactoring goals
            
        Returns:
            Dict[str, Any]: Refactored code and changes
        """
        # Generate prompt
        prompt = self.templates["code_refactor"].format(
            language=self.language,
            code=code,
            goals="\n".join(f"- {goal}" for goal in goals),
            style_guide=json.dumps(self.style_guide, indent=2)
        )
        
        # Get response from LLM
        response = await self.llm.agenerate(prompt)
        
        # Extract refactored code
        refactored_code = self._extract_code(response)
        
        # Analyze changes
        analysis = {
            "original": self._analyze_code(code),
            "refactored": self._analyze_code(refactored_code)
        }
        
        return {
            "original_code": code,
            "refactored_code": refactored_code,
            "analysis": analysis,
            "changes": self._diff_code(code, refactored_code)
        }
        
    def _extract_code(self, text: str) -> str:
        """Extract code blocks from text.
        
        Args:
            text: Text containing code
            
        Returns:
            str: Extracted code
        """
        # Look for code blocks
        code_blocks = re.findall(r"```.*?\n(.*?)```", text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
            
        # If no code blocks, try to extract based on indentation
        lines = text.split("\n")
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.startswith("    ") or line.startswith("\t"):
                code_lines.append(line.strip())
                in_code = True
            elif in_code and not line.strip():
                code_lines.append("")
            elif in_code:
                break
                
        return "\n".join(code_lines) if code_lines else text
        
    def _analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code structure and complexity.
        
        Args:
            code: Code to analyze
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            if self.code_analyzer and self.language.lower() == "python":
                tree = self.code_analyzer.parse(code)
                return {
                    "num_functions": len([
                        node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef)
                    ]),
                    "num_classes": len([
                        node for node in ast.walk(tree)
                        if isinstance(node, ast.ClassDef)
                    ]),
                    "complexity": self._calculate_complexity(tree)
                }
        except Exception as e:
            self.log_error(e, {
                "code_length": len(code),
                "operation": "code_analysis"
            })
            
        # Fallback to basic analysis
        return {
            "num_lines": len(code.split("\n")),
            "avg_line_length": sum(len(line) for line in code.split("\n")) / len(code.split("\n"))
        }
        
    def _check_style(self, code: str) -> List[Dict[str, Any]]:
        """Check code against style guide.
        
        Args:
            code: Code to check
            
        Returns:
            List[Dict[str, Any]]: Style violations
        """
        violations = []
        
        # Check line length
        max_length = self.style_guide.get("max_line_length", 80)
        for i, line in enumerate(code.split("\n"), 1):
            if len(line) > max_length:
                violations.append({
                    "line": i,
                    "type": "line_length",
                    "message": f"Line exceeds {max_length} characters"
                })
                
        # Add more style checks based on language
        
        return violations
        
    def _calculate_metrics(self, code: str) -> Dict[str, float]:
        """Calculate code quality metrics.
        
        Args:
            code: Code to analyze
            
        Returns:
            Dict[str, float]: Quality metrics
        """
        lines = code.split("\n")
        return {
            "lines_of_code": len(lines),
            "comment_ratio": len([l for l in lines if l.strip().startswith("#")]) / len(lines),
            "blank_line_ratio": len([l for l in lines if not l.strip()]) / len(lines)
        }
        
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity.
        
        Args:
            tree: AST tree
            
        Returns:
            int: Complexity score
        """
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
                
        return complexity
        
    def _diff_code(self, original: str, refactored: str) -> List[Dict[str, Any]]:
        """Generate diff between original and refactored code.
        
        Args:
            original: Original code
            refactored: Refactored code
            
        Returns:
            List[Dict[str, Any]]: List of changes
        """
        import difflib
        
        differ = difflib.Differ()
        diff = list(differ.compare(
            original.splitlines(keepends=True),
            refactored.splitlines(keepends=True)
        ))
        
        changes = []
        for i, line in enumerate(diff):
            if line.startswith(("+ ", "- ", "? ")):
                changes.append({
                    "type": line[0],
                    "line_number": i,
                    "content": line[2:].strip()
                })
                
        return changes