import unittest
import os
from llm_models.base_llm import BaseLLM

class TestLLMIntegration(unittest.TestCase):
    def setUp(self):
        self.llm = BaseLLM()

    def test_llm_initialization(self):
        """Test that LLM can be initialized"""
        self.assertIsNotNone(self.llm.get_llm())
        # Check if we're using the expected backend
        expected_backend = "openai" if os.getenv("OPENAI_API_KEY") else "llamafile"
        self.assertEqual(self.llm.backend_type, expected_backend)

    def test_simple_prompt(self):
        """Test that LLM can generate a response"""
        prompt = "<|system|>You are a helpful AI assistant.</|system|>\n<|user|>What is 2+2?</|user|>\n<|assistant|>"
        response = self.llm.generate(prompt)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        print(f"\nResponse from {self.llm.backend_type} backend: {response}")

    def test_complex_prompt(self):
        """Test that LLM can handle a more complex prompt"""
        prompt = """<|system|>You are a helpful AI assistant.</|system|>
<|user|>Consider the following Python code:
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

What are the potential performance issues with this implementation?</|user|>
<|assistant|>"""
        response = self.llm.generate(prompt)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        print(f"\nResponse from {self.llm.backend_type} backend: {response}")

if __name__ == '__main__':
    unittest.main()