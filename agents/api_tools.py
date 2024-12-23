"""External API integration tools for agents."""
import os
from typing import List, Dict, Any, Optional, Union
import json
import aiohttp
import asyncio
from datetime import datetime
from pathlib import Path
from functools import wraps
import yaml
from rich.console import Console

from utils.logging_util import setup_logger

logger = setup_logger(__name__)
console = Console()

def require_api_key(func):
    """Decorator to check for required API key."""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        api_name = func.__name__.split('_')[0]
        if not self.has_api_key(api_name):
            raise ValueError(f"Missing API key for {api_name}")
        return await func(self, *args, **kwargs)
    return wrapper

class APITools:
    """Tools for interacting with external APIs."""
    
    def __init__(self, config_dir: str = "config/apis"):
        """Initialize API tools.
        
        Args:
            config_dir: Directory for API configurations
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configurations
        self.configs = self._load_configs()
        
        # Initialize session
        self.session = None
        
        # Track API usage
        self.usage_logs: Dict[str, List[Dict[str, Any]]] = {}
        
    async def __aenter__(self):
        """Enter async context."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        if self.session:
            await self.session.close()
            
    def has_api_key(self, api_name: str) -> bool:
        """Check if API key is configured.
        
        Args:
            api_name: Name of API
            
        Returns:
            bool: True if API key exists
        """
        return (
            api_name in self.configs and
            "api_key" in self.configs[api_name]
        )
        
    @require_api_key
    async def finance_get_stock_data(self,
                                   symbol: str,
                                   interval: str = "1d") -> Dict[str, Any]:
        """Get stock market data.
        
        Args:
            symbol: Stock symbol
            interval: Time interval
            
        Returns:
            Dict containing stock data
        """
        config = self.configs["finance"]
        url = f"{config['base_url']}/stock/{symbol}"
        
        params = {
            "interval": interval,
            "apikey": config["api_key"]
        }
        
        async with self.session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            self._log_api_call("finance", "get_stock_data", {
                "symbol": symbol,
                "interval": interval
            })
            
            return data
            
    @require_api_key
    async def finance_get_company_info(self,
                                     symbol: str) -> Dict[str, Any]:
        """Get company information.
        
        Args:
            symbol: Company symbol
            
        Returns:
            Dict containing company info
        """
        config = self.configs["finance"]
        url = f"{config['base_url']}/company/{symbol}"
        
        params = {
            "apikey": config["api_key"]
        }
        
        async with self.session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            self._log_api_call("finance", "get_company_info", {
                "symbol": symbol
            })
            
            return data
            
    @require_api_key
    async def marketing_get_social_metrics(self,
                                         platform: str,
                                         account_id: str) -> Dict[str, Any]:
        """Get social media metrics.
        
        Args:
            platform: Social media platform
            account_id: Account ID
            
        Returns:
            Dict containing social metrics
        """
        config = self.configs["marketing"]
        url = f"{config['base_url']}/social/{platform}/metrics"
        
        params = {
            "account_id": account_id,
            "api_key": config["api_key"]
        }
        
        async with self.session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            self._log_api_call("marketing", "get_social_metrics", {
                "platform": platform,
                "account_id": account_id
            })
            
            return data
            
    @require_api_key
    async def marketing_analyze_campaign(self,
                                       campaign_id: str) -> Dict[str, Any]:
        """Analyze marketing campaign.
        
        Args:
            campaign_id: Campaign ID
            
        Returns:
            Dict containing campaign analysis
        """
        config = self.configs["marketing"]
        url = f"{config['base_url']}/campaign/{campaign_id}/analyze"
        
        params = {
            "api_key": config["api_key"]
        }
        
        async with self.session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            self._log_api_call("marketing", "analyze_campaign", {
                "campaign_id": campaign_id
            })
            
            return data
            
    @require_api_key
    async def tech_analyze_code(self,
                              code: str,
                              language: str) -> Dict[str, Any]:
        """Analyze code for quality and security.
        
        Args:
            code: Code to analyze
            language: Programming language
            
        Returns:
            Dict containing code analysis
        """
        config = self.configs["tech"]
        url = f"{config['base_url']}/code/analyze"
        
        data = {
            "code": code,
            "language": language,
            "api_key": config["api_key"]
        }
        
        async with self.session.post(url, json=data) as response:
            response.raise_for_status()
            result = await response.json()
            
            self._log_api_call("tech", "analyze_code", {
                "language": language,
                "code_length": len(code)
            })
            
            return result
            
    @require_api_key
    async def tech_check_dependencies(self,
                                    dependencies: List[str]) -> Dict[str, Any]:
        """Check dependencies for security issues.
        
        Args:
            dependencies: List of dependencies
            
        Returns:
            Dict containing dependency analysis
        """
        config = self.configs["tech"]
        url = f"{config['base_url']}/dependencies/check"
        
        data = {
            "dependencies": dependencies,
            "api_key": config["api_key"]
        }
        
        async with self.session.post(url, json=data) as response:
            response.raise_for_status()
            result = await response.json()
            
            self._log_api_call("tech", "check_dependencies", {
                "num_dependencies": len(dependencies)
            })
            
            return result
            
    def _load_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load API configurations."""
        configs = {}
        
        for file in self.config_dir.glob("*.yaml"):
            try:
                with open(file) as f:
                    api_name = file.stem
                    configs[api_name] = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error loading config {file}: {str(e)}")
                
        return configs
        
    def _log_api_call(self,
                      api_name: str,
                      endpoint: str,
                      params: Dict[str, Any]):
        """Log API call.
        
        Args:
            api_name: Name of API
            endpoint: API endpoint
            params: Call parameters
        """
        if api_name not in self.usage_logs:
            self.usage_logs[api_name] = []
            
        self.usage_logs[api_name].append({
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "params": params
        })
        
    def show_api_usage(self, api_name: Optional[str] = None):
        """Show API usage statistics.
        
        Args:
            api_name: Optional API name to filter
        """
        table = Table(title="API Usage Statistics")
        table.add_column("API")
        table.add_column("Endpoint")
        table.add_column("Calls")
        table.add_column("Last Used")
        
        apis = [api_name] if api_name else self.usage_logs.keys()
        
        for api in apis:
            if api not in self.usage_logs:
                continue
                
            # Group by endpoint
            endpoint_stats = {}
            for call in self.usage_logs[api]:
                endpoint = call["endpoint"]
                if endpoint not in endpoint_stats:
                    endpoint_stats[endpoint] = {
                        "calls": 0,
                        "last_used": call["timestamp"]
                    }
                endpoint_stats[endpoint]["calls"] += 1
                endpoint_stats[endpoint]["last_used"] = max(
                    endpoint_stats[endpoint]["last_used"],
                    call["timestamp"]
                )
                
            # Add to table
            for endpoint, stats in endpoint_stats.items():
                table.add_row(
                    api,
                    endpoint,
                    str(stats["calls"]),
                    stats["last_used"]
                )
                
        console.print(table)
        
    def export_usage_logs(self, output_file: str):
        """Export usage logs to file.
        
        Args:
            output_file: Output file path
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(self.usage_logs, f, indent=2)
            logger.info(f"Exported usage logs to {output_file}")
        except Exception as e:
            logger.error(f"Error exporting logs: {str(e)}")