from typing import Dict, Any, Optional, List, Union
import re
import json
import hashlib
import jwt
from datetime import datetime, timedelta
import aiohttp
from utils.logging_util import LoggerMixin

class SecurityManager(LoggerMixin):
    """Manager for system security and input filtering."""
    
    def __init__(self,
                 config_path: str = "config/security",
                 jwt_secret: Optional[str] = None,
                 token_expiry: int = 24):  # hours
        """Initialize security manager.
        
        Args:
            config_path: Path to security config
            jwt_secret: JWT secret key
            token_expiry: Token expiry in hours
        """
        super().__init__()
        self.config_path = config_path
        self.jwt_secret = jwt_secret or self._generate_secret()
        self.token_expiry = timedelta(hours=token_expiry)
        
        # Load security rules
        self.rules = self._load_security_rules()
        
        # Track blocked attempts
        self.blocked_attempts: Dict[str, List[Dict[str, Any]]] = {}
        
        # API rate limiting
        self.api_calls: Dict[str, List[datetime]] = {}
        self.rate_limits = {
            "default": {"calls": 100, "period": 3600},  # 100 calls per hour
            "high_priority": {"calls": 1000, "period": 3600}
        }
        
    def _generate_secret(self) -> str:
        """Generate JWT secret key.
        
        Returns:
            str: Secret key
        """
        return hashlib.sha256(
            datetime.now().isoformat().encode()
        ).hexdigest()
        
    def _load_security_rules(self) -> Dict[str, Any]:
        """Load security rules.
        
        Returns:
            Dict[str, Any]: Security rules
        """
        return {
            "prompt_filters": {
                "blocked_terms": [
                    "hack", "exploit", "vulnerability",
                    "password", "credential", "token"
                ],
                "max_length": 1000,
                "required_patterns": [
                    r"^[a-zA-Z0-9\s\.,\?!-]+$"  # Basic alphanumeric + punctuation
                ]
            },
            "api_security": {
                "allowed_domains": [
                    "api.openai.com",
                    "huggingface.co",
                    "github.com"
                ],
                "required_headers": ["Authorization"],
                "blocked_methods": ["DELETE", "PUT"]
            },
            "data_validation": {
                "max_file_size": 10 * 1024 * 1024,  # 10MB
                "allowed_extensions": [".txt", ".json", ".csv", ".py"],
                "content_types": [
                    "text/plain",
                    "application/json",
                    "text/csv",
                    "text/x-python"
                ]
            }
        }
        
    def generate_token(self,
                      user_id: str,
                      permissions: List[str]) -> str:
        """Generate JWT token.
        
        Args:
            user_id: User identifier
            permissions: User permissions
            
        Returns:
            str: JWT token
        """
        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "exp": datetime.utcnow() + self.token_expiry,
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(
            payload,
            self.jwt_secret,
            algorithm="HS256"
        )
        
        # Log token generation
        self.log_event(
            "token_generated",
            {
                "user_id": user_id,
                "permissions": permissions
            }
        )
        
        return token
        
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token.
        
        Args:
            token: JWT token
            
        Returns:
            Optional[Dict[str, Any]]: Token payload or None
        """
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=["HS256"]
            )
            return payload
        except jwt.ExpiredSignatureError:
            self.log_event(
                "token_expired",
                {"token": token[:10] + "..."}
            )
            return None
        except jwt.InvalidTokenError as e:
            self.log_error(e, {"token": token[:10] + "..."})
            return None
            
    def check_permission(self,
                        token: str,
                        required_permission: str) -> bool:
        """Check if token has permission.
        
        Args:
            token: JWT token
            required_permission: Required permission
            
        Returns:
            bool: Whether permission is granted
        """
        payload = self.validate_token(token)
        if not payload:
            return False
            
        return required_permission in payload.get("permissions", [])
        
    def filter_prompt(self,
                     prompt: str,
                     user_id: Optional[str] = None) -> Optional[str]:
        """Filter and validate prompt.
        
        Args:
            prompt: Input prompt
            user_id: Optional user identifier
            
        Returns:
            Optional[str]: Filtered prompt or None if blocked
        """
        rules = self.rules["prompt_filters"]
        
        # Check length
        if len(prompt) > rules["max_length"]:
            self._record_blocked_attempt(
                user_id,
                "prompt_too_long",
                {"length": len(prompt)}
            )
            return None
            
        # Check blocked terms
        for term in rules["blocked_terms"]:
            if term.lower() in prompt.lower():
                self._record_blocked_attempt(
                    user_id,
                    "blocked_term",
                    {"term": term}
                )
                return None
                
        # Check patterns
        for pattern in rules["required_patterns"]:
            if not re.match(pattern, prompt):
                self._record_blocked_attempt(
                    user_id,
                    "invalid_pattern",
                    {"pattern": pattern}
                )
                return None
                
        return prompt
        
    def validate_api_request(self,
                           url: str,
                           method: str,
                           headers: Dict[str, str]) -> bool:
        """Validate API request.
        
        Args:
            url: Request URL
            method: HTTP method
            headers: Request headers
            
        Returns:
            bool: Whether request is valid
        """
        rules = self.rules["api_security"]
        
        # Check domain
        domain = url.split("/")[2]
        if domain not in rules["allowed_domains"]:
            self.log_event(
                "blocked_domain",
                {"domain": domain}
            )
            return False
            
        # Check method
        if method in rules["blocked_methods"]:
            self.log_event(
                "blocked_method",
                {"method": method}
            )
            return False
            
        # Check headers
        for header in rules["required_headers"]:
            if header not in headers:
                self.log_event(
                    "missing_header",
                    {"header": header}
                )
                return False
                
        return True
        
    def validate_file(self,
                     filename: str,
                     size: int,
                     content_type: str) -> bool:
        """Validate file upload.
        
        Args:
            filename: File name
            size: File size in bytes
            content_type: File content type
            
        Returns:
            bool: Whether file is valid
        """
        rules = self.rules["data_validation"]
        
        # Check size
        if size > rules["max_file_size"]:
            self.log_event(
                "file_too_large",
                {
                    "filename": filename,
                    "size": size
                }
            )
            return False
            
        # Check extension
        ext = os.path.splitext(filename)[1].lower()
        if ext not in rules["allowed_extensions"]:
            self.log_event(
                "invalid_extension",
                {
                    "filename": filename,
                    "extension": ext
                }
            )
            return False
            
        # Check content type
        if content_type not in rules["content_types"]:
            self.log_event(
                "invalid_content_type",
                {
                    "filename": filename,
                    "content_type": content_type
                }
            )
            return False
            
        return True
        
    def check_rate_limit(self,
                        user_id: str,
                        priority: str = "default") -> bool:
        """Check API rate limit.
        
        Args:
            user_id: User identifier
            priority: Rate limit priority
            
        Returns:
            bool: Whether request is allowed
        """
        now = datetime.now()
        limit = self.rate_limits[priority]
        
        # Initialize user calls
        if user_id not in self.api_calls:
            self.api_calls[user_id] = []
            
        # Clean old calls
        self.api_calls[user_id] = [
            t for t in self.api_calls[user_id]
            if (now - t).total_seconds() < limit["period"]
        ]
        
        # Check limit
        if len(self.api_calls[user_id]) >= limit["calls"]:
            self.log_event(
                "rate_limit_exceeded",
                {
                    "user_id": user_id,
                    "priority": priority
                }
            )
            return False
            
        # Add call
        self.api_calls[user_id].append(now)
        return True
        
    def _record_blocked_attempt(self,
                              user_id: Optional[str],
                              reason: str,
                              details: Dict[str, Any]):
        """Record blocked attempt.
        
        Args:
            user_id: Optional user identifier
            reason: Block reason
            details: Block details
        """
        if not user_id:
            return
            
        if user_id not in self.blocked_attempts:
            self.blocked_attempts[user_id] = []
            
        self.blocked_attempts[user_id].append({
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "details": details
        })
        
        # Log attempt
        self.log_event(
            "blocked_attempt",
            {
                "user_id": user_id,
                "reason": reason,
                **details
            }
        )
        
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics.
        
        Returns:
            Dict[str, Any]: Security statistics
        """
        stats = {
            "blocked_attempts": {
                "total": sum(
                    len(attempts)
                    for attempts in self.blocked_attempts.values()
                ),
                "by_reason": {}
            },
            "api_usage": {
                "total_calls": sum(
                    len(calls)
                    for calls in self.api_calls.values()
                ),
                "active_users": len(self.api_calls)
            }
        }
        
        # Count blocked attempts by reason
        for attempts in self.blocked_attempts.values():
            for attempt in attempts:
                reason = attempt["reason"]
                if reason not in stats["blocked_attempts"]["by_reason"]:
                    stats["blocked_attempts"]["by_reason"][reason] = 0
                stats["blocked_attempts"]["by_reason"][reason] += 1
                
        return stats