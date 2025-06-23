import asyncio
from aiohttp import web

class FakeOpenHandsServer:
    """Minimal async server mocking OpenHands API."""

    def __init__(self):
        self.app = web.Application()
        self.app.router.add_post('/docker/run', self.run_container)
        self.runner = web.AppRunner(self.app)
        self.site = None
        self.port = None

    async def run_container(self, request: web.Request) -> web.Response:
        data = await request.json()
        fail = data.get('fail')
        if fail == 'unauthorized':
            return web.Response(status=401)
        if fail == 'error':
            return web.Response(status=500)
        return web.json_response({'container_id': 'mock123', 'status': 'running', 'logs': 'started'})

    async def __aenter__(self):
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, 'localhost', 0)
        await self.site.start()
        self.port = self.site._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.runner.cleanup()

    @property
    def url(self) -> str:
        return f'http://localhost:{self.port}'
