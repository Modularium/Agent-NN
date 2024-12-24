"""Input filtering and security checks."""
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path
import yaml
from rich.console import Console

from utils.logging_util import setup_logger

logger = setup_logger(__name__)
console = Console()

class SecurityFilter:
    """Filter for checking and sanitizing inputs."""
    
    def __init__(self, config_dir: str = "config/security"):
        """Initialize security filter.
        
        Args:
            config_dir: Directory for security configurations
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configurations
        self.blocked_patterns = self._load_patterns("blocked_patterns.yaml")
        self.sensitive_patterns = self._load_patterns("sensitive_patterns.yaml")
        self.allowed_domains = self._load_list("allowed_domains.yaml")
        self.api_rules = self._load_rules("api_rules.yaml")
        
        # Track violations
        self.violations: List[Dict[str, Any]] = []
        
    def check_input(self,
                   text: str,
                   context: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """Check if input is safe.
        
        Args:
            text: Input text to check
            context: Optional context information
            
        Returns:
            Tuple of (is_safe, reason)
        """
        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern["regex"], text, re.IGNORECASE):
                self._log_violation("blocked_pattern", {
                    "pattern": pattern["name"],
                    "text": text,
                    "context": context
                })
                return False, f"Contains blocked pattern: {pattern['name']}"
                
        # Check sensitive patterns
        for pattern in self.sensitive_patterns:
            if re.search(pattern["regex"], text, re.IGNORECASE):
                self._log_violation("sensitive_pattern", {
                    "pattern": pattern["name"],
                    "text": text,
                    "context": context
                })
                return False, f"Contains sensitive information: {pattern['name']}"
                
        return True, ""
        
    def check_api_access(self,
                        api_name: str,
                        endpoint: str,
                        params: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if API access is allowed.
        
        Args:
            api_name: Name of API
            endpoint: API endpoint
            params: API parameters
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        # Check if API is configured
        if api_name not in self.api_rules:
            return False, f"Unknown API: {api_name}"
            
        rules = self.api_rules[api_name]
        
        # Check if endpoint is allowed
        if endpoint not in rules["allowed_endpoints"]:
            self._log_violation("api_access", {
                "api": api_name,
                "endpoint": endpoint,
                "params": params
            })
            return False, f"Endpoint not allowed: {endpoint}"
            
        # Check parameter rules
        endpoint_rules = rules["endpoints"].get(endpoint, {})
        for param, value in params.items():
            # Check required parameters
            if param in endpoint_rules.get("required", []):
                if not value:
                    return False, f"Missing required parameter: {param}"
                    
            # Check parameter patterns
            if param in endpoint_rules.get("patterns", {}):
                pattern = endpoint_rules["patterns"][param]
                if not re.match(pattern, str(value)):
                    self._log_violation("api_parameter", {
                        "api": api_name,
                        "endpoint": endpoint,
                        "parameter": param,
                        "value": value
                    })
                    return False, f"Invalid parameter value: {param}"
                    
        return True, ""
        
    def check_url(self, url: str) -> Tuple[bool, str]:
        """Check if URL is allowed.
        
        Args:
            url: URL to check
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        # Extract domain
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
        except Exception:
            return False, "Invalid URL format"
            
        # Check against allowed domains
        if not any(
            domain.endswith(allowed_domain)
            for allowed_domain in self.allowed_domains
        ):
            self._log_violation("url_access", {
                "url": url,
                "domain": domain
            })
            return False, f"Domain not allowed: {domain}"
            
        return True, ""
        
    def sanitize_input(self, text: str) -> str:
        """Sanitize input text.
        
        Args:
            text: Input text
            
        Returns:
            Sanitized text
        """
        # Remove potential HTML/script tags
        text = re.sub(r'<[^>]*>', '', text)
        
        # Remove potential SQL injection patterns
        text = re.sub(r'[\'";\-]', '', text)
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32)
        
        return text.strip()
        
    def get_violations(self,
                      violation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get security violations.
        
        Args:
            violation_type: Optional type to filter by
            
        Returns:
            List of violations
        """
        if violation_type:
            return [v for v in self.violations if v["type"] == violation_type]
        return self.violations
        
    def _load_patterns(self, filename: str) -> List[Dict[str, Any]]:
        """Load regex patterns from file.
        
        Args:
            filename: Pattern file name
            
        Returns:
            List of pattern dictionaries
        """
        file_path = self.config_dir / filename
        if not file_path.exists():
            return self._create_default_patterns(filename)
            
        try:
            with open(file_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading patterns from {filename}: {str(e)}")
            return []
            
    def _load_list(self, filename: str) -> List[str]:
        """Load list from file.
        
        Args:
            filename: List file name
            
        Returns:
            List of strings
        """
        file_path = self.config_dir / filename
        if not file_path.exists():
            return self._create_default_list(filename)
            
        try:
            with open(file_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading list from {filename}: {str(e)}")
            return []
            
    def _load_rules(self, filename: str) -> Dict[str, Any]:
        """Load API rules from file.
        
        Args:
            filename: Rules file name
            
        Returns:
            Dict containing rules
        """
        file_path = self.config_dir / filename
        if not file_path.exists():
            return self._create_default_rules(filename)
            
        try:
            with open(file_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading rules from {filename}: {str(e)}")
            return {}
            
    def _create_default_patterns(self, filename: str) -> List[Dict[str, Any]]:
        """Create default pattern file.
        
        Args:
            filename: Pattern file name
            
        Returns:
            List of default patterns
        """
        if filename == "blocked_patterns.yaml":
            patterns = [
                {
                    "name": "sql_injection",
                    "regex": r"(?i)(select|insert|update|delete|drop|union|exec)\s+",
                    "description": "SQL injection attempt"
                },
                {
                    "name": "xss_script",
                    "regex": r"(?i)<script[^>]*>",
                    "description": "XSS script tag"
                },
                {
                    "name": "path_traversal",
                    "regex": r"\.\.\/|\.\.\\",
                    "description": "Path traversal attempt"
                }
            ]
        else:  # sensitive_patterns.yaml
            patterns = [
                {
                    "name": "credit_card",
                    "regex": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
                    "description": "Credit card number"
                },
                {
                    "name": "social_security",
                    "regex": r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",
                    "description": "Social security number"
                },
                {
                    "name": "api_key",
                    "regex": r"(?i)(api[_-]?key|access[_-]?token)['\"]?\s*[:=]\s*['\"]\w+",
                    "description": "API key or access token"
                }
            ]
            
        # Save patterns
        file_path = self.config_dir / filename
        with open(file_path, 'w') as f:
            yaml.dump(patterns, f)
            
        return patterns
        
    def _create_default_list(self, filename: str) -> List[str]:
        """Create default list file.
        
        Args:
            filename: List file name
            
        Returns:
            List of default values
        """
        if filename == "allowed_domains.yaml":
            domains = [
                "github.com",
                "api.openai.com",
                "huggingface.co",
                "amazonaws.com"
            ]
        else:
            domains = []
            
        # Save list
        file_path = self.config_dir / filename
        with open(file_path, 'w') as f:
            yaml.dump(domains, f)
            
        return domains
        
    def _create_default_rules(self, filename: str) -> Dict[str, Any]:
        """Create default rules file.
        
        Args:
            filename: Rules file name
            
        Returns:
            Dict containing default rules
        """
        rules = {
            "finance": {
                "allowed_endpoints": ["stock", "company"],
                "endpoints": {
                    "stock": {
                        "required": ["symbol"],
                        "patterns": {
                            "symbol": r"^[A-Z]{1,5}$",
                            "interval": r"^[1-9]\d?[dhm]$"
                        }
                    },
                    "company": {
                        "required": ["symbol"],
                        "patterns": {
                            "symbol": r"^[A-Z]{1,5}$"
                        }
                    }
                }
            },
            "marketing": {
                "allowed_endpoints": ["social", "campaign"],
                "endpoints": {
                    "social": {
                        "required": ["platform", "account_id"],
                        "patterns": {
                            "platform": r"^(twitter|facebook|instagram)$"
                        }
                    }
                }
            }
        }
        
        # Save rules
        file_path = self.config_dir / filename
        with open(file_path, 'w') as f:
            yaml.dump(rules, f)
            
        return rules
        
    def _log_violation(self,
                      violation_type: str,
                      details: Dict[str, Any]):
        """Log security violation.
        
        Args:
            violation_type: Type of violation
            details: Violation details
        """
        violation = {
            "type": violation_type,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        
        self.violations.append(violation)
        logger.warning(f"Security violation: {violation_type}")
        
    def show_violations(self,
                       violation_type: Optional[str] = None):
        """Show security violations.
        
        Args:
            violation_type: Optional type to filter by
        """
        violations = self.get_violations(violation_type)
        
        table = Table(title="Security Violations")
        table.add_column("Type")
        table.add_column("Timestamp")
        table.add_column("Details")
        
        for violation in violations:
            table.add_row(
                violation["type"],
                violation["timestamp"],
                str(violation["details"])
            )
            
        console.print(table)
        
    def export_violations(self, output_file: str):
        """Export violations to file.
        
        Args:
            output_file: Output file path
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(self.violations, f, indent=2)
            logger.info(f"Exported violations to {output_file}")
        except Exception as e:
            logger.error(f"Error exporting violations: {str(e)}")