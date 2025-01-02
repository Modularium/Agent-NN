import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from agents.software_dev.python_agent import PythonDevAgent
from agents.software_dev.typescript_agent import TypeScriptDevAgent
from langchain.schema import Document

class TestPythonDevAgent(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch('utils.logging_util.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        
        # Mock LLM
        self.llm_patcher = patch('agents.software_dev.base_dev_agent.SpecializedLLM')
        self.mock_llm = self.llm_patcher.start()
        
        # Set up mock LLM instance
        self.mock_llm_instance = AsyncMock()
        self.mock_llm_instance.agenerate.return_value = "```python\ndef test():\n    pass\n```"
        self.mock_llm.return_value = self.mock_llm_instance
        
        # Initialize agent
        self.agent = PythonDevAgent()
        
    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        self.llm_patcher.stop()
        
    async def test_code_generation(self):
        """Test Python code generation."""
        result = await self.agent.generate_code(
            task_description="Create a simple function",
            requirements=["Return None"]
        )
        
        # Check result structure
        self.assertIn("code", result)
        self.assertIn("analysis", result)
        self.assertEqual(result["language"], "python")
        
        # Verify LLM was called
        self.mock_llm_instance.agenerate.assert_called_once()
        
    async def test_code_review(self):
        """Test Python code review."""
        code = "def test():\n    pass\n"
        result = await self.agent.review_code(code)
        
        # Check result structure
        self.assertIn("review", result)
        self.assertIn("analysis", result)
        self.assertIn("style_violations", result)
        self.assertIn("metrics", result)
        
    async def test_code_formatting(self):
        """Test Python code formatting."""
        code = "def test(): pass"
        result = await self.agent.format_code(code)
        
        # Check result structure
        self.assertIn("original_code", result)
        self.assertIn("formatted_code", result)
        self.assertIn("changes", result)
        
    def test_code_analysis(self):
        """Test Python code analysis."""
        code = """
        def test():
            pass
            
        class Test:
            def method(self):
                pass
        """
        
        analysis = self.agent._analyze_code(code)
        
        # Check analysis results
        self.assertIn("num_functions", analysis)
        self.assertIn("num_classes", analysis)
        self.assertIn("docstring_coverage", analysis)

class TestTypeScriptDevAgent(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch('utils.logging_util.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        
        # Mock LLM
        self.llm_patcher = patch('agents.software_dev.base_dev_agent.SpecializedLLM')
        self.mock_llm = self.llm_patcher.start()
        
        # Set up mock LLM instance
        self.mock_llm_instance = AsyncMock()
        self.mock_llm_instance.agenerate.return_value = "```typescript\nfunction test(): void {\n}\n```"
        self.mock_llm.return_value = self.mock_llm_instance
        
        # Mock subprocess
        self.subprocess_patcher = patch('agents.software_dev.typescript_agent.subprocess')
        self.mock_subprocess = self.subprocess_patcher.start()
        
        # Initialize agent
        self.agent = TypeScriptDevAgent()
        
    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        self.llm_patcher.stop()
        self.subprocess_patcher.stop()
        
    async def test_code_generation(self):
        """Test TypeScript code generation."""
        result = await self.agent.generate_code(
            task_description="Create a simple function",
            requirements=["Return void"]
        )
        
        # Check result structure
        self.assertIn("code", result)
        self.assertIn("analysis", result)
        self.assertEqual(result["language"], "typescript")
        
        # Verify LLM was called
        self.mock_llm_instance.agenerate.assert_called_once()
        
    async def test_code_review(self):
        """Test TypeScript code review."""
        code = "function test(): void {\n}\n"
        result = await self.agent.review_code(code)
        
        # Check result structure
        self.assertIn("review", result)
        self.assertIn("analysis", result)
        self.assertIn("style_violations", result)
        self.assertIn("metrics", result)
        
    async def test_code_formatting(self):
        """Test TypeScript code formatting."""
        code = "function test():void{}"
        
        # Mock prettier output
        self.mock_subprocess.run.return_value.stdout = "function test(): void {\n}\n"
        
        result = await self.agent.format_code(code)
        
        # Check result structure
        self.assertIn("original_code", result)
        self.assertIn("formatted_code", result)
        self.assertIn("changes", result)
        
    def test_type_analysis(self):
        """Test TypeScript type analysis."""
        code = """
        interface Test {
            prop: string;
        }
        
        type Alias = string | number;
        
        enum Status {
            Active,
            Inactive
        }
        """
        
        analysis = self.agent._analyze_code(code)
        
        # Check analysis results
        self.assertIn("types", analysis)
        self.assertIn("interfaces", analysis["types"])
        self.assertIn("enums", analysis["types"])
        
    def test_import_analysis(self):
        """Test TypeScript import analysis."""
        code = """
        import { Component } from '@angular/core';
        import type { Config } from './config';
        import * as utils from './utils';
        import defaultExport from 'module';
        """
        
        imports = self.agent._analyze_imports(code)
        
        # Check import categories
        self.assertIn("named", imports)
        self.assertIn("type", imports)
        self.assertIn("namespace", imports)
        self.assertIn("default", imports)

if __name__ == '__main__':
    unittest.main()