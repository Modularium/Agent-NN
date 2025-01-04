from typing import Dict, Any, Optional, List, Union
import asyncio
from playwright.async_api import async_playwright, Browser, Page
import json
from datetime import datetime
import re
from utils.logging_util import LoggerMixin

class JSRenderer(LoggerMixin):
    """Renderer for JavaScript-enabled web content."""
    
    def __init__(self,
                 headless: bool = True,
                 timeout: int = 30000,
                 wait_until: str = "networkidle",
                 max_retries: int = 3,
                 concurrent_pages: int = 5):
        """Initialize JavaScript renderer.
        
        Args:
            headless: Run browser in headless mode
            timeout: Page load timeout in milliseconds
            wait_until: When to consider page loaded
            max_retries: Maximum retry attempts
            concurrent_pages: Maximum concurrent pages
        """
        super().__init__()
        self.headless = headless
        self.timeout = timeout
        self.wait_until = wait_until
        self.max_retries = max_retries
        self.concurrent_pages = concurrent_pages
        
        # Playwright resources
        self.playwright = None
        self.browser: Optional[Browser] = None
        
        # Semaphore for concurrent pages
        self.page_semaphore = asyncio.Semaphore(concurrent_pages)
        
        # Error tracking
        self.error_counts: Dict[str, int] = {}
        
    async def initialize(self):
        """Initialize Playwright browser."""
        if not self.browser:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless
            )
            
    async def cleanup(self):
        """Clean up resources."""
        if self.browser:
            await self.browser.close()
            self.browser = None
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
            
    async def render_page(self,
                         url: str,
                         selectors: Optional[Dict[str, str]] = None,
                         wait_for: Optional[Union[str, List[str]]] = None,
                         javascript: Optional[str] = None) -> Dict[str, Any]:
        """Render page with JavaScript and extract content.
        
        Args:
            url: URL to render
            selectors: Optional CSS selectors to extract
            wait_for: Selectors to wait for before extraction
            javascript: Optional JavaScript to execute
            
        Returns:
            Dict[str, Any]: Extracted content
        """
        async with self.page_semaphore:
            await self.initialize()
            
            # Create new page
            page = await self.browser.new_page()
            
            try:
                # Configure timeout
                page.set_default_timeout(self.timeout)
                
                # Navigate with retries
                content = None
                last_error = None
                
                for attempt in range(self.max_retries):
                    try:
                        # Navigate to page
                        await page.goto(
                            url,
                            wait_until=self.wait_until,
                            timeout=self.timeout
                        )
                        
                        # Wait for specific elements
                        if wait_for:
                            if isinstance(wait_for, str):
                                wait_for = [wait_for]
                            for selector in wait_for:
                                await page.wait_for_selector(
                                    selector,
                                    timeout=self.timeout
                                )
                                
                        # Execute custom JavaScript
                        if javascript:
                            await page.evaluate(javascript)
                            
                        # Extract content
                        content = await self._extract_content(page, selectors)
                        break
                        
                    except Exception as e:
                        last_error = e
                        if attempt < self.max_retries - 1:
                            # Track error
                            error_key = f"{url}:{type(e).__name__}"
                            self.error_counts[error_key] = (
                                self.error_counts.get(error_key, 0) + 1
                            )
                            
                            # Log retry
                            self.log_event(
                                "render_retry",
                                {
                                    "url": url,
                                    "attempt": attempt + 1,
                                    "error": str(e)
                                }
                            )
                            
                            # Exponential backoff
                            await asyncio.sleep(2 ** attempt)
                            
                if not content:
                    raise last_error or Exception("Failed to render page")
                    
                # Log success
                self.log_event(
                    "render_success",
                    {
                        "url": url,
                        "content_size": len(str(content))
                    }
                )
                
                return content
                
            except Exception as e:
                self.log_error(e, {
                    "url": url,
                    "operation": "render_page"
                })
                raise
                
            finally:
                await page.close()
                
    async def _extract_content(self,
                             page: Page,
                             selectors: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Extract content from rendered page.
        
        Args:
            page: Playwright page
            selectors: CSS selectors to extract
            
        Returns:
            Dict[str, Any]: Extracted content
        """
        content = {}
        
        # Get full page content if no selectors
        if not selectors:
            content["html"] = await page.content()
            content["text"] = await page.evaluate(
                "document.body.innerText"
            )
            return content
            
        # Extract content using selectors
        for key, selector in selectors.items():
            try:
                elements = await page.query_selector_all(selector)
                if elements:
                    # Get text content
                    texts = []
                    for element in elements:
                        text = await element.text_content()
                        if text:
                            texts.append(text.strip())
                            
                    # Store single value or list
                    if len(texts) == 1:
                        content[key] = texts[0]
                    else:
                        content[key] = texts
                        
            except Exception as e:
                self.log_error(e, {
                    "selector": selector,
                    "operation": "extract_content"
                })
                
        return content
        
    async def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics.
        
        Returns:
            Dict[str, Any]: Error statistics
        """
        stats = {
            "total_errors": sum(self.error_counts.values()),
            "unique_errors": len(self.error_counts),
            "error_types": {},
            "most_frequent": sorted(
                self.error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
        
        # Group by error type
        for error_key, count in self.error_counts.items():
            url, error_type = error_key.split(":", 1)
            if error_type not in stats["error_types"]:
                stats["error_types"][error_type] = 0
            stats["error_types"][error_type] += count
            
        return stats
        
    def clear_error_stats(self):
        """Clear error statistics."""
        self.error_counts.clear()
        self.log_event("error_stats_cleared", {})