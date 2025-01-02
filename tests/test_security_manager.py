import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import jwt
from managers.security_manager import SecurityManager

class TestSecurityManager(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch('utils.logging_util.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        
        # Initialize manager
        self.manager = SecurityManager(
            jwt_secret="test_secret",
            token_expiry=1  # 1 hour
        )
        
    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        
    def test_token_generation(self):
        """Test JWT token generation."""
        # Generate token
        token = self.manager.generate_token(
            "test_user",
            ["read", "write"]
        )
        
        # Validate token
        payload = self.manager.validate_token(token)
        
        # Check payload
        self.assertEqual(payload["user_id"], "test_user")
        self.assertEqual(payload["permissions"], ["read", "write"])
        
    def test_token_expiry(self):
        """Test token expiration."""
        # Generate token
        token = self.manager.generate_token(
            "test_user",
            ["read"]
        )
        
        # Move time forward
        future_time = datetime.utcnow() + timedelta(hours=2)
        with patch('jwt.decode') as mock_decode:
            mock_decode.side_effect = jwt.ExpiredSignatureError
            
            # Validate expired token
            payload = self.manager.validate_token(token)
            self.assertIsNone(payload)
            
    def test_permission_check(self):
        """Test permission checking."""
        # Generate token with permissions
        token = self.manager.generate_token(
            "test_user",
            ["read", "write"]
        )
        
        # Check permissions
        self.assertTrue(
            self.manager.check_permission(token, "read")
        )
        self.assertTrue(
            self.manager.check_permission(token, "write")
        )
        self.assertFalse(
            self.manager.check_permission(token, "admin")
        )
        
    def test_prompt_filtering(self):
        """Test prompt filtering."""
        # Test valid prompt
        valid_prompt = "This is a valid prompt."
        filtered = self.manager.filter_prompt(valid_prompt)
        self.assertEqual(filtered, valid_prompt)
        
        # Test blocked terms
        blocked_prompt = "Please hack the system."
        filtered = self.manager.filter_prompt(blocked_prompt)
        self.assertIsNone(filtered)
        
        # Test length limit
        long_prompt = "x" * 2000
        filtered = self.manager.filter_prompt(long_prompt)
        self.assertIsNone(filtered)
        
    def test_api_validation(self):
        """Test API request validation."""
        # Test valid request
        valid = self.manager.validate_api_request(
            "https://api.openai.com/v1/chat",
            "POST",
            {"Authorization": "Bearer token"}
        )
        self.assertTrue(valid)
        
        # Test blocked domain
        invalid = self.manager.validate_api_request(
            "https://malicious.com/api",
            "GET",
            {"Authorization": "Bearer token"}
        )
        self.assertFalse(invalid)
        
        # Test blocked method
        invalid = self.manager.validate_api_request(
            "https://api.openai.com/v1/chat",
            "DELETE",
            {"Authorization": "Bearer token"}
        )
        self.assertFalse(invalid)
        
    def test_file_validation(self):
        """Test file validation."""
        # Test valid file
        valid = self.manager.validate_file(
            "test.txt",
            1024,
            "text/plain"
        )
        self.assertTrue(valid)
        
        # Test large file
        invalid = self.manager.validate_file(
            "large.txt",
            20 * 1024 * 1024,
            "text/plain"
        )
        self.assertFalse(invalid)
        
        # Test invalid extension
        invalid = self.manager.validate_file(
            "script.exe",
            1024,
            "application/octet-stream"
        )
        self.assertFalse(invalid)
        
    def test_rate_limiting(self):
        """Test rate limiting."""
        user_id = "test_user"
        
        # Make allowed number of requests
        for _ in range(100):
            allowed = self.manager.check_rate_limit(user_id)
            self.assertTrue(allowed)
            
        # Next request should be blocked
        blocked = self.manager.check_rate_limit(user_id)
        self.assertFalse(blocked)
        
        # Test high priority
        allowed = self.manager.check_rate_limit(
            user_id,
            priority="high_priority"
        )
        self.assertTrue(allowed)
        
    def test_blocked_attempts_tracking(self):
        """Test blocked attempts tracking."""
        user_id = "test_user"
        
        # Generate some blocked attempts
        self.manager.filter_prompt(
            "hack the system",
            user_id
        )
        self.manager.filter_prompt(
            "x" * 2000,
            user_id
        )
        
        # Check stats
        stats = self.manager.get_security_stats()
        self.assertEqual(
            stats["blocked_attempts"]["total"],
            2
        )
        self.assertEqual(
            len(stats["blocked_attempts"]["by_reason"]),
            2
        )
        
    def test_security_stats(self):
        """Test security statistics."""
        # Generate some activity
        user_id = "test_user"
        
        # API calls
        for _ in range(5):
            self.manager.check_rate_limit(user_id)
            
        # Blocked attempts
        self.manager.filter_prompt(
            "hack the system",
            user_id
        )
        
        # Get stats
        stats = self.manager.get_security_stats()
        
        # Check stats structure
        self.assertIn("blocked_attempts", stats)
        self.assertIn("api_usage", stats)
        self.assertEqual(stats["api_usage"]["active_users"], 1)
        self.assertEqual(stats["api_usage"]["total_calls"], 5)

if __name__ == '__main__':
    unittest.main()