import asyncio
from typing import Any, Optional

import httpx
from fastapi import HTTPException


class ServiceConnector:
    """Helper for internal service requests with retry and timeout."""

    def __init__(self, base_url: str, timeout: float = 10.0, retries: int = 2) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.client = httpx.AsyncClient()

    async def request(self, method: str, path: str, json: Optional[dict] = None) -> Any:
        url = f"{self.base_url}{path}"
        for attempt in range(self.retries + 1):
            try:
                resp = await self.client.request(
                    method, url, json=json, timeout=self.timeout
                )
                resp.raise_for_status()
                return resp.json()
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                if attempt >= self.retries:
                    raise HTTPException(
                        status_code=503, detail="ServiceUnavailable"
                    ) from exc
                await asyncio.sleep(0.2 * (attempt + 1))

    async def get(self, path: str) -> Any:
        return await self.request("GET", path)

    async def post(self, path: str, json: dict) -> Any:
        return await self.request("POST", path, json=json)
