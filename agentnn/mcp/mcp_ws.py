"""WebSocket server for streaming MCP events."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState


class MCPWebSocketServer:
    """Manage connections and broadcast session events."""

    def __init__(self) -> None:
        self.router = APIRouter()
        self.connections: List[WebSocket] = []
        self._register()

    def _register(self) -> None:
        @self.router.websocket("/ws/session/{sid}")
        async def session_ws(websocket: WebSocket, sid: str) -> None:
            await websocket.accept()
            self.connections.append(websocket)
            try:
                while True:
                    # keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.connections.remove(websocket)

    async def broadcast(self, event: Dict[str, Any]) -> None:
        """Send an event to all connected clients."""
        for ws in list(self.connections):
            if ws.application_state == WebSocketState.CONNECTED:
                await ws.send_json(event)
            else:
                self.connections.remove(ws)


ws_server = MCPWebSocketServer()
