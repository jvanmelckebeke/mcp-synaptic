"""SSE client for connecting to MCP Synaptic servers."""

import asyncio
import json
from datetime import datetime
from typing import Any, Callable, Dict, Optional
from urllib.parse import urljoin

try:
    import aiohttp
except ImportError:
    aiohttp = None

from ..config.logging import LoggerMixin
from ..core.exceptions import ConnectionError, SSEError
from .events import Event, EventType


class SSEClient(LoggerMixin):
    """Client for connecting to SSE endpoints."""

    def __init__(self, base_url: str, client_id: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.client_id = client_id
        self.session: Optional[aiohttp.ClientSession] = None
        self._event_handlers: Dict[EventType, Callable] = {}
        self._running = False
        self._connect_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize the SSE client."""
        if aiohttp is None:
            raise SSEError("aiohttp not available. Install with: pip install aiohttp")

        self.session = aiohttp.ClientSession()
        self.logger.info("SSE client initialized", base_url=self.base_url)

    async def close(self) -> None:
        """Close the SSE client."""
        await self.disconnect()
        
        if self.session:
            await self.session.close()
            self.session = None
        
        self.logger.info("SSE client closed")

    def on(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """Register an event handler."""
        self._event_handlers[event_type] = handler
        self.logger.debug("Event handler registered", event_type=event_type.value)

    def off(self, event_type: EventType) -> None:
        """Remove an event handler."""
        self._event_handlers.pop(event_type, None)
        self.logger.debug("Event handler removed", event_type=event_type.value)

    async def connect(self, endpoint: str = "/events") -> None:
        """Connect to the SSE endpoint."""
        if not self.session:
            raise SSEError("Client not initialized")

        if self._running:
            self.logger.warning("Client already connected")
            return

        url = urljoin(self.base_url, endpoint)
        params = {}
        
        if self.client_id:
            params['client_id'] = self.client_id

        self._running = True
        self._connect_task = asyncio.create_task(self._connection_loop(url, params))
        
        self.logger.info("SSE client connecting", url=url)

    async def disconnect(self) -> None:
        """Disconnect from the SSE endpoint."""
        self._running = False
        
        if self._connect_task and not self._connect_task.done():
            self._connect_task.cancel()
            try:
                await self._connect_task
            except asyncio.CancelledError:
                pass

        self.logger.info("SSE client disconnected")

    async def _connection_loop(self, url: str, params: Dict[str, Any]) -> None:
        """Main connection loop for handling SSE events."""
        while self._running:
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status != 200:
                        raise ConnectionError(f"SSE connection failed: HTTP {response.status}")

                    self.logger.info("SSE connection established")
                    
                    async for line in response.content:
                        if not self._running:
                            break

                        line = line.decode('utf-8').strip()
                        if not line:
                            continue

                        await self._handle_sse_line(line)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("SSE connection error", error=str(e))
                
                if self._running:
                    self.logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)

    async def _handle_sse_line(self, line: str) -> None:
        """Handle a single SSE line."""
        try:
            if line.startswith('data: '):
                data_str = line[6:]  # Remove 'data: ' prefix
                
                try:
                    data = json.loads(data_str)
                    await self._process_event_data(data)
                except json.JSONDecodeError as e:
                    self.logger.warning("Failed to decode SSE data", data=data_str, error=str(e))

        except Exception as e:
            self.logger.error("Error handling SSE line", line=line, error=str(e))

    async def _process_event_data(self, data: Dict[str, Any]) -> None:
        """Process event data and call appropriate handlers."""
        try:
            # Extract event type from data or use a default
            event_type_str = data.get('event_type', 'unknown')
            
            try:
                event_type = EventType(event_type_str)
            except ValueError:
                self.logger.warning("Unknown event type", event_type=event_type_str)
                return

            # Create event object
            event = Event(
                event_type=event_type,
                data=data,
                timestamp=datetime.fromisoformat(data.get('timestamp', datetime.utcnow().isoformat()))
            )

            # Call registered handler
            handler = self._event_handlers.get(event_type)
            if handler:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    self.logger.error(
                        "Error in event handler",
                        event_type=event_type.value,
                        error=str(e)
                    )

            self.logger.debug("Event processed", event_type=event_type.value)

        except Exception as e:
            self.logger.error("Error processing event data", data=data, error=str(e))

    @property
    def is_connected(self) -> bool:
        """Check if the client is connected."""
        return self._running and self._connect_task is not None and not self._connect_task.done()

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "connected": self.is_connected,
            "base_url": self.base_url,
            "client_id": self.client_id,
            "registered_handlers": len(self._event_handlers),
        }