from __future__ import annotations

import signal
from fastapi import FastAPI
import uvicorn


def run_service(app: FastAPI, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run a FastAPI app with graceful shutdown handlers."""
    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)

    def _shutdown(*_: object) -> None:
        server.should_exit = True

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    server.run()
