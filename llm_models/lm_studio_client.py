"""LM-Studio API client implementation."""
import json
from typing import Dict, Any, Optional
import aiohttp
import logging

logger = logging.getLogger(__name__)

class LMStudioClient:
    """Client for interacting with LM-Studio API."""
    
    def __init__(self, 
                 base_url: str = "http://192.168.0.247:1234",
                 temperature: float = 0.7,
                 max_tokens: int = 2048):
        """Initialize LM-Studio client.
        
        Args:
            base_url: Base URL for LM-Studio API
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    async def generate_response(self, prompt: str) -> str:
        """Generate response from LM-Studio API.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response text
            
        Raises:
            Exception: If API call fails
        """
        try:
            logger.debug(f"Sending prompt to LM-Studio: {prompt}")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens
                    }
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"API call failed with status {response.status}: {error_text}")
                        raise Exception(f"API call failed with status {response.status}: {error_text}")
                        
                    result = await response.json()
                    logger.debug(f"Raw API response: {result}")
                    
                    response_text = result["choices"][0]["message"]["content"]
                    logger.debug(f"Extracted response: {response_text}")
                    
                    # Try to parse as JSON if it looks like JSON
                    if response_text.strip().startswith('{') or response_text.strip().startswith('['):
                        try:
                            json_data = json.loads(response_text)
                            logger.debug(f"Successfully parsed as JSON: {json_data}")
                            return json.dumps(json_data)  # Return normalized JSON string
                        except json.JSONDecodeError:
                            logger.debug("Response looks like JSON but failed to parse")
                            pass
                            
                    return response_text
                    
        except Exception as e:
            logger.error(f"Error calling LM-Studio API: {str(e)}")
            raise