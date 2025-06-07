"""Server-Sent Events server implementation."""

import asyncio
import weakref
from datetime import datetime
from typing import Dict, List, Optional, Set
from uuid import uuid4

try:
    from sse_starlette import EventSourceResponse
except ImportError:
    EventSourceResponse = None

from ..config.logging import LoggerMixin
from ..config.settings import Settings
from ..core.exceptions import SSEError
from .events import Event, HeartbeatEvent, ConnectionEvent


class SSEClient:
    """Represents an SSE client connection."""
    
    def __init__(self, client_id: str, queue: asyncio.Queue):
        self.client_id = client_id
        self.queue = queue
        self.connected_at = datetime.utcnow()
        self.last_heartbeat = datetime.utcnow()
        self.subscriptions: Set[str] = set()  # Event types this client subscribes to

    def subscribe(self, event_type: str) -> None:
        """Subscribe to an event type."""
        self.subscriptions.add(event_type)

    def unsubscribe(self, event_type: str) -> None:
        """Unsubscribe from an event type."""
        self.subscriptions.discard(event_type)

    def is_subscribed(self, event_type: str) -> bool:
        """Check if client is subscribed to an event type."""
        return event_type in self.subscriptions or not self.subscriptions  # Subscribe to all if empty


class SSEServer(LoggerMixin):
    """Server-Sent Events server for real-time communication."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.clients: Dict[str, SSEClient] = {}
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False

    async def initialize(self) -> None:
        """Initialize the SSE server."""
        if EventSourceResponse is None:
            raise SSEError("sse-starlette not available. Install with: pip install sse-starlette")

        self._running = True
        
        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        self.logger.info(
            "SSE server initialized",
            heartbeat_interval=self.settings.SSE_HEARTBEAT_INTERVAL,
            max_connections=self.settings.SSE_MAX_CONNECTIONS
        )

    async def shutdown(self) -> None:
        """Shutdown the SSE server."""
        self._running = False
        
        # Cancel heartbeat task
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Disconnect all clients
        for client_id in list(self.clients.keys()):
            await self._disconnect_client(client_id)

        self.logger.info("SSE server shutdown")

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat events to all clients."""
        while self._running:
            try:
                if self.clients:
                    heartbeat_event = HeartbeatEvent()
                    await self.broadcast_event(heartbeat_event)
                
                await asyncio.sleep(self.settings.SSE_HEARTBEAT_INTERVAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in heartbeat loop", error=str(e))
                await asyncio.sleep(self.settings.SSE_HEARTBEAT_INTERVAL)

    async def create_event_stream(self, client_id: Optional[str] = None):
        """Create an SSE event stream for a client."""
        if len(self.clients) >= self.settings.SSE_MAX_CONNECTIONS:
            raise SSEError("Maximum SSE connections reached")

        client_id = client_id or str(uuid4())
        queue = asyncio.Queue()
        
        client = SSEClient(client_id, queue)
        self.clients[client_id] = client

        # Send connection event
        connection_event = ConnectionEvent(client_id, "connected")
        await self.send_to_client(client_id, connection_event)

        self.logger.info("SSE client connected", client_id=client_id)

        try:
            async def event_generator():
                """Generate events for the SSE stream."""
                try:
                    while self._running and client_id in self.clients:
                        # Wait for next event with timeout
                        try:
                            event = await asyncio.wait_for(queue.get(), timeout=1.0)
                            yield event.to_sse_format()
                        except asyncio.TimeoutError:
                            # Send periodic ping to keep connection alive
                            continue
                        
                except asyncio.CancelledError:
                    pass
                finally:
                    await self._disconnect_client(client_id)

            return EventSourceResponse(event_generator())

        except Exception as e:
            await self._disconnect_client(client_id)
            raise SSEError(f"Failed to create event stream: {e}")

    async def send_to_client(self, client_id: str, event: Event) -> bool:
        """Send an event to a specific client."""
        client = self.clients.get(client_id)
        if not client:
            self.logger.warning("Attempt to send to non-existent client", client_id=client_id)
            return False

        try:
            # Check if client is subscribed to this event type
            if not client.is_subscribed(event.event_type.value):
                return True  # Successfully ignored

            await client.queue.put(event)
            
            self.logger.debug(
                "Event sent to client",
                client_id=client_id,
                event_type=event.event_type.value
            )
            return True

        except Exception as e:
            self.logger.error(
                "Failed to send event to client",
                client_id=client_id,
                event_type=event.event_type.value,
                error=str(e)
            )
            return False

    async def broadcast_event(self, event: Event, exclude_clients: Optional[List[str]] = None) -> int:
        """Broadcast an event to all connected clients."""
        exclude_clients = exclude_clients or []
        sent_count = 0

        for client_id in list(self.clients.keys()):
            if client_id not in exclude_clients:
                if await self.send_to_client(client_id, event):
                    sent_count += 1

        self.logger.debug(
            "Event broadcasted",
            event_type=event.event_type.value,
            sent_to=sent_count,
            total_clients=len(self.clients)
        )

        return sent_count

    async def subscribe_client(self, client_id: str, event_types: List[str]) -> bool:
        """Subscribe a client to specific event types."""
        client = self.clients.get(client_id)
        if not client:
            return False

        for event_type in event_types:
            client.subscribe(event_type)

        self.logger.info(
            "Client subscribed to events",
            client_id=client_id,
            event_types=event_types
        )
        return True

    async def unsubscribe_client(self, client_id: str, event_types: List[str]) -> bool:
        """Unsubscribe a client from specific event types."""
        client = self.clients.get(client_id)
        if not client:
            return False

        for event_type in event_types:
            client.unsubscribe(event_type)

        self.logger.info(
            "Client unsubscribed from events",
            client_id=client_id,
            event_types=event_types
        )
        return True

    async def _disconnect_client(self, client_id: str) -> None:
        """Disconnect a client and clean up resources."""
        client = self.clients.pop(client_id, None)
        if client:
            # Send disconnect event to other clients
            disconnect_event = ConnectionEvent(client_id, "disconnected")
            await self.broadcast_event(disconnect_event, exclude_clients=[client_id])

            self.logger.info("SSE client disconnected", client_id=client_id)

    def get_connected_clients(self) -> List[Dict]:
        """Get information about connected clients."""
        return [
            {
                "client_id": client.client_id,
                "connected_at": client.connected_at.isoformat(),
                "last_heartbeat": client.last_heartbeat.isoformat(),
                "subscriptions": list(client.subscriptions),
            }
            for client in self.clients.values()
        ]

    def get_stats(self) -> Dict:
        """Get SSE server statistics."""
        return {
            "connected_clients": len(self.clients),
            "max_connections": self.settings.SSE_MAX_CONNECTIONS,
            "heartbeat_interval": self.settings.SSE_HEARTBEAT_INTERVAL,
            "running": self._running,
        }