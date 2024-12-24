"""Tests for model setup functionality."""
import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from scripts.setup_local_models import (
    download_file,
    setup_llamafile,
    setup_lmstudio,
    verify_setup
)
from config.llm_config import LLAMAFILE_CONFIG

class TestModelSetup(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.original_models_dir = LLAMAFILE_CONFIG["models_dir"]
        LLAMAFILE_CONFIG["models_dir"] = self.test_dir
        
    def tearDown(self):
        """Clean up test environment."""
        # Restore original config and remove test directory
        LLAMAFILE_CONFIG["models_dir"] = self.original_models_dir
        shutil.rmtree(self.test_dir)
        
    @patch('requests.get')
    def test_file_download(self, mock_get):
        """Test file download functionality."""
        # Mock response with content
        mock_response = MagicMock()
        mock_response.headers = {'content-length': '100'}
        mock_response.iter_content.return_value = [b'test data']
        mock_get.return_value = mock_response
        
        # Test download
        test_file = os.path.join(self.test_dir, 'test.file')
        download_file('http://test.url', test_file)
        
        # Verify file was created
        self.assertTrue(os.path.exists(test_file))
        with open(test_file, 'rb') as f:
            self.assertEqual(f.read(), b'test data')
            
    @patch('scripts.setup_local_models.download_file')
    def test_llamafile_setup(self, mock_download):
        """Test Llamafile model setup."""
        # Test with valid model
        result = setup_llamafile('llama-2-7b')
        self.assertTrue(result)
        mock_download.assert_called_once()
        
        # Test with invalid model
        result = setup_llamafile('invalid-model')
        self.assertFalse(result)
        
    @patch('builtins.print')
    def test_lmstudio_setup(self, mock_print):
        """Test LM Studio setup instructions."""
        result = setup_lmstudio()
        self.assertTrue(result)
        
        # Verify instructions were printed
        mock_print.assert_called()
        instructions = ' '.join(str(call.args[0]) for call in mock_print.call_args_list)
        self.assertIn('lmstudio.ai', instructions.lower())
        self.assertIn('local server', instructions.lower())
        
    @patch('requests.get')
    def test_setup_verification(self, mock_get):
        """Test setup verification."""
        # Create dummy model file
        model_name = 'llama-2-7b'
        model_path = os.path.join(
            self.test_dir,
            LLAMAFILE_CONFIG["models"][model_name]["filename"]
        )
        with open(model_path, 'w') as f:
            f.write('dummy model')
        os.chmod(model_path, 0o755)
        
        # Mock LM Studio server response
        mock_get.return_value = MagicMock(status_code=200)
        
        # Capture verification output
        with patch('builtins.print') as mock_print:
            verify_setup()
            
            # Verify output
            output = ' '.join(str(call.args[0]) for call in mock_print.call_args_list)
            self.assertIn(model_name, output)
            self.assertIn('installed', output)
            self.assertIn('running', output)
            
    def test_error_handling(self):
        """Test error handling in setup process."""
        # Test download error
        with patch('scripts.setup_local_models.download_file', 
                  side_effect=Exception('Download failed')):
            result = setup_llamafile('llama-2-7b')
            self.assertFalse(result)
            
        # Test invalid model path
        with patch('os.chmod', side_effect=OSError('Permission denied')):
            result = setup_llamafile('llama-2-7b')
            self.assertFalse(result)
            
    def test_concurrent_downloads(self):
        """Test downloading multiple models concurrently."""
        # This is a placeholder for potential concurrent download tests
        # In practice, you might want to implement async downloads
        pass
        
    def test_model_verification(self):
        """Test model file verification."""
        model_name = 'llama-2-7b'
        model_path = os.path.join(
            self.test_dir,
            LLAMAFILE_CONFIG["models"][model_name]["filename"]
        )
        
        # Test with missing file
        self.assertFalse(os.path.exists(model_path))
        
        # Test with existing file
        with open(model_path, 'w') as f:
            f.write('dummy model')
        os.chmod(model_path, 0o755)
        
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.access(model_path, os.X_OK))
        
    def test_environment_variables(self):
        """Test environment variable handling."""
        # Test custom models directory
        custom_dir = tempfile.mkdtemp()
        try:
            with patch.dict(os.environ, {'LLAMAFILE_MODELS_DIR': custom_dir}):
                # Reload config to pick up new environment variable
                import importlib
                import config.llm_config
                importlib.reload(config.llm_config)
                
                self.assertEqual(
                    config.llm_config.LLAMAFILE_CONFIG["models_dir"],
                    custom_dir
                )
        finally:
            shutil.rmtree(custom_dir)
            
    def test_model_compatibility(self):
        """Test model compatibility checks."""
        # This would test system requirements, available memory, etc.
        # For now, just a placeholder
        pass

if __name__ == '__main__':
    unittest.main()