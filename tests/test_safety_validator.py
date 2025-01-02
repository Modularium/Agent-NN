import unittest
from agents.software_dev.safety_validator import CodeSafetyValidator

class TestCodeSafetyValidator(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.validator = CodeSafetyValidator()
        
    def test_python_validation(self):
        """Test Python code validation."""
        # Test dangerous builtins
        code = """
        def dangerous():
            eval("print('hello')")
            exec("x = 1")
            __import__('os').system('ls')
        """
        violations = self.validator.validate_python(code)
        self.assertTrue(any(v["type"] == "dangerous_builtin" for v in violations))
        
        # Test dangerous imports
        code = """
        import os
        import sys
        from subprocess import run
        """
        violations = self.validator.validate_python(code)
        self.assertTrue(any(v["type"] == "dangerous_import" for v in violations))
        
        # Test safe code
        code = """
        def safe():
            x = 1
            y = 2
            return x + y
        """
        violations = self.validator.validate_python(code)
        self.assertEqual(len(violations), 0)
        
    def test_javascript_validation(self):
        """Test JavaScript code validation."""
        # Test dangerous patterns
        code = """
        eval('alert("hello")');
        setTimeout(() => {}, 1000);
        new Function('return true');
        document.write('test');
        """
        violations = self.validator.validate_javascript(code)
        self.assertTrue(len(violations) > 0)
        
        # Test safe code
        code = """
        function safe() {
            const x = 1;
            const y = 2;
            return x + y;
        }
        """
        violations = self.validator.validate_javascript(code)
        self.assertEqual(len(violations), 0)
        
    def test_html_validation(self):
        """Test HTML code validation."""
        # Test dangerous patterns
        code = """
        <html>
            <script>alert('hello')</script>
            <div onclick="alert('click')">Click me</div>
            <a href="javascript:void(0)">Link</a>
        </html>
        """
        violations = self.validator.validate_html(code)
        self.assertTrue(any(v["type"] == "inline_script" for v in violations))
        self.assertTrue(any(v["type"] == "event_handler" for v in violations))
        
        # Test safe code
        code = """
        <html>
            <div>Safe content</div>
            <a href="https://example.com">Safe link</a>
        </html>
        """
        violations = self.validator.validate_html(code)
        self.assertEqual(len(violations), 0)
        
    def test_css_validation(self):
        """Test CSS code validation."""
        # Test dangerous patterns
        code = """
        .dangerous {
            background: url('unsafe://example.com/image.jpg');
            behavior: expression(alert('xss'));
        }
        @import 'unsafe://example.com/style.css';
        """
        violations = self.validator.validate_css(code)
        self.assertTrue(any(v["type"] == "suspicious_url" for v in violations))
        self.assertTrue(any(v["type"] == "dangerous_expression" for v in violations))
        
        # Test safe code
        code = """
        .safe {
            background: url('https://example.com/image.jpg');
            color: #000;
            padding: 10px;
        }
        """
        violations = self.validator.validate_css(code)
        self.assertEqual(len(violations), 0)
        
    def test_safe_subset_extraction(self):
        """Test safe subset extraction."""
        # Python code
        code = """
        import os
        import math
        
        def dangerous():
            os.system('ls')
            
        def safe():
            return math.sqrt(16)
        """
        safe_subset = self.validator.get_safe_subset(code, "python")
        self.assertNotIn("os.system", safe_subset)
        self.assertIn("math.sqrt", safe_subset)
        
        # JavaScript code
        code = """
        function dangerous() {
            eval('alert()');
        }
        
        function safe() {
            return Math.sqrt(16);
        }
        """
        safe_subset = self.validator.get_safe_subset(code, "javascript")
        self.assertNotIn("eval", safe_subset)
        self.assertIn("Math.sqrt", safe_subset)
        
    def test_strict_validation(self):
        """Test strict validation mode."""
        code = """
        # Python code with medium severity violation
        x = input('Enter value: ')
        """
        
        # Strict mode should reject
        self.assertFalse(self.validator.is_code_safe(code, "python", strict=True))
        
        # Non-strict mode should accept
        self.assertTrue(self.validator.is_code_safe(code, "python", strict=False))

if __name__ == '__main__':
    unittest.main()