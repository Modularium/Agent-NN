"""Tests for local LLM integration."""
import os
import unittest
from unittest.mock import patch, MagicMock
import requests
import subprocess
from llm_models.llm_backend import (
    LLMBackendType,
    LLMBackendManager,
    LMStudioLLM,
    LlamafileLLM
)
from config.llm_config import (
    LLAMAFILE_CONFIG,
    LMSTUDIO_CONFIG,
    get_model_path,
    get_model_config
)

class TestLocalLLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create test models directory
        cls.test_models_dir = os.path.join(os.path.dirname(__file__), "test_models")
        os.makedirs(cls.test_models_dir, exist_ok=True)
        
        # Create dummy Llamafile for testing
        cls.test_llamafile = os.path.join(cls.test_models_dir, "test.llamafile")
        with open(cls.test_llamafile, "w") as f:
            f.write("#!/bin/bash\necho 'Test response'")
        os.chmod(cls.test_llamafile, 0o755)
        
    def setUp(self):
        """Set up each test."""
        self.backend_manager = LLMBackendManager()
        
    def test_backend_switching(self):
        """Test switching between different LLM backends."""
        # Test initial state
        self.assertEqual(self.backend_manager.current_backend, LLMBackendType.OPENAI)
        
        # Test switching to LM Studio
        self.backend_manager.set_backend(LLMBackendType.LMSTUDIO)
        self.assertEqual(self.backend_manager.current_backend, LLMBackendType.LMSTUDIO)
        
        # Test switching to Llamafile
        self.backend_manager.set_backend(LLMBackendType.LLAMAFILE)
        self.assertEqual(self.backend_manager.current_backend, LLMBackendType.LLAMAFILE)
        
    @patch('requests.post')
    def test_lmstudio_integration(self, mock_post):
        """Test LM Studio integration."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"text": "Test response"}]
        }
        mock_post.return_value = mock_response
        
        # Initialize LM Studio LLM
        llm = LMStudioLLM()
        
        # Test generation
        response = llm._call("Test prompt")
        self.assertEqual(response, "Test response")
        
        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]
        self.assertIn("prompt", call_args["json"])
        self.assertEqual(call_args["json"]["prompt"], "Test prompt")
        
    def test_llamafile_integration(self):
        """Test Llamafile integration."""
        # Initialize Llamafile LLM with test file
        llm = LlamafileLLM(
            model_path=self.test_llamafile,
            port=8081
        )
        
        try:
            # Start server
            llm.start_server()
            
            # Verify server process
            self.assertIsNotNone(llm.process)
            self.assertTrue(llm.process.pid > 0)
            
        finally:
            # Clean up
            llm.stop_server()
            
    def test_model_config_loading(self):
        """Test model configuration loading."""
        # Test Llamafile config
        llama_config = get_model_config("llamafile", "llama-2-7b")
        self.assertIn("max_tokens", llama_config)
        self.assertIn("temperature", llama_config)
        
        # Test LM Studio config
        lmstudio_config = get_model_config("lmstudio", "local-model")
        self.assertIn("max_tokens", lmstudio_config)
        self.assertIn("temperature", lmstudio_config)
        
    def test_model_path_resolution(self):
        """Test model path resolution."""
        # Test valid model
        model_name = "llama-2-7b"
        path = get_model_path(model_name)
        self.assertTrue(os.path.dirname(path))
        self.assertTrue(path.endswith(LLAMAFILE_CONFIG["models"][model_name]["filename"]))
        
        # Test invalid model
        with self.assertRaises(ValueError):
            get_model_path("invalid-model")
            
    @patch('requests.get')
    def test_lmstudio_server_check(self, mock_get):
        """Test LM Studio server availability check."""
        # Test server running
        mock_get.return_value = MagicMock(status_code=200)
        self.assertTrue(self._check_lmstudio_server())
        
        # Test server not running
        mock_get.side_effect = requests.ConnectionError()
        self.assertFalse(self._check_lmstudio_server())
        
    def _check_lmstudio_server(self):
        """Helper to check LM Studio server."""
        try:
            response = requests.get(LMSTUDIO_CONFIG["endpoint_url"])
            return response.status_code == 200
        except:
            return False
            
    def test_error_handling(self):
        """Test error handling in local LLM integration."""
        # Test invalid model path
        with self.assertRaises(FileNotFoundError):
            LlamafileLLM(model_path="/nonexistent/path")
            
        # Test invalid port
        with self.assertRaises(ValueError):
            LlamafileLLM(model_path=self.test_llamafile, port=-1)
            
    def test_concurrent_model_usage(self):
        """Test using multiple local models concurrently."""
        llm1 = LlamafileLLM(
            model_path=self.test_llamafile,
            port=8081
        )
        llm2 = LlamafileLLM(
            model_path=self.test_llamafile,
            port=8082
        )
        
        try:
            # Start both servers
            llm1.start_server()
            llm2.start_server()
            
            # Verify both are running
            self.assertIsNotNone(llm1.process)
            self.assertIsNotNone(llm2.process)
            self.assertNotEqual(llm1.process.pid, llm2.process.pid)
            
        finally:
            # Clean up
            llm1.stop_server()
            llm2.stop_server()
            
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove test models directory
        import shutil
        shutil.rmtree(cls.test_models_dir)

if __name__ == '__main__':
    unittest.main()