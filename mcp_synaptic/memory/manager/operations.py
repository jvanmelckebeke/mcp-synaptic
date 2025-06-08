"""Memory CRUD operations handler."""

from datetime import datetime, UTC
from typing import Dict, Optional

from ...config.settings import Settings
from ...core.exceptions import MemoryError, MemoryExpiredError, MemoryNotFoundError
from ..storage import MemoryStorage
from ...models.memory import ExpirationPolicy, Memory, MemoryType


class MemoryOperations:
    """Handles CRUD operations for memory management."""

    def __init__(self, storage: MemoryStorage, settings: Settings, logger):
        self.storage = storage
        self.settings = settings
        self.logger = logger

    async def add(
        self,
        key: str,
        data: Dict,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        ttl_seconds: Optional[int] = None,
        expiration_policy: ExpirationPolicy = ExpirationPolicy.ABSOLUTE,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict] = None,
    ) -> Memory:
        """Add a new memory or update existing one."""
        try:
            # Use default TTL if not specified
            if ttl_seconds is None:
                ttl_seconds = self._get_default_ttl(memory_type)

            # Create memory object
            memory = Memory(
                key=key,
                data=data,
                memory_type=memory_type,
                expiration_policy=expiration_policy,
                ttl_seconds=ttl_seconds,
                tags=tags or {},
                metadata=metadata or {},
            )

            # Set expiration based on policy
            memory.update_expiration(ttl_seconds)

            # Store the memory
            await self.storage.store(memory)

            self.logger.info(
                "Memory added",
                key=key,
                memory_type=memory_type.value,
                ttl_seconds=ttl_seconds,
                expires_at=memory.expires_at.isoformat() if memory.expires_at else None
            )

            return memory

        except Exception as e:
            self.logger.error("Failed to add memory", key=key, error=str(e))
            raise MemoryError(f"Failed to add memory '{key}': {e}")

    async def get(self, key: str, touch: bool = True) -> Optional[Memory]:
        """Retrieve a memory by key."""
        try:
            memory = await self.storage.retrieve(key)
            
            if not memory:
                return None

            # Check if expired
            if memory.is_expired:
                # Remove expired memory
                await self.storage.delete(key)
                self.logger.debug("Removed expired memory", key=key)
                raise MemoryExpiredError(key)

            # Update access information if requested
            if touch:
                memory.touch()
                await self.storage.store(memory)

            self.logger.debug("Memory retrieved", key=key, access_count=memory.access_count)
            return memory

        except MemoryExpiredError:
            raise
        except Exception as e:
            self.logger.error("Failed to get memory", key=key, error=str(e))
            raise MemoryError(f"Failed to get memory '{key}': {e}")

    async def delete(self, key: str) -> bool:
        """Delete a memory by key."""
        try:
            deleted = await self.storage.delete(key)
            
            if deleted:
                self.logger.info("Memory deleted", key=key)
            else:
                self.logger.debug("Memory not found for deletion", key=key)
            
            return deleted

        except Exception as e:
            self.logger.error("Failed to delete memory", key=key, error=str(e))
            raise MemoryError(f"Failed to delete memory '{key}': {e}")

    async def update(
        self,
        key: str,
        data: Optional[Dict] = None,
        extend_ttl: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[Memory]:
        """Update an existing memory."""
        try:
            memory = await self.get(key, touch=False)
            if not memory:
                raise MemoryNotFoundError(key)

            # Update fields
            if data is not None:
                memory.data = data
            if tags is not None:
                memory.tags.update(tags)
            if metadata is not None:
                memory.metadata.update(metadata)

            # Update timestamps
            memory.updated_at = datetime.now(UTC)

            # Extend TTL if requested
            if extend_ttl is not None:
                memory.ttl_seconds = extend_ttl
                memory.update_expiration(extend_ttl)

            # Store updated memory
            await self.storage.store(memory)

            self.logger.info("Memory updated", key=key)
            return memory

        except (MemoryNotFoundError, MemoryExpiredError):
            raise
        except Exception as e:
            self.logger.error("Failed to update memory", key=key, error=str(e))
            raise MemoryError(f"Failed to update memory '{key}': {e}")

    def _get_default_ttl(self, memory_type: MemoryType) -> int:
        """Get default TTL for a memory type."""
        defaults = {
            MemoryType.EPHEMERAL: 300,    # 5 minutes
            MemoryType.SHORT_TERM: self.settings.DEFAULT_MEMORY_TTL_SECONDS,
            MemoryType.LONG_TERM: 86400 * 7,  # 1 week
            MemoryType.PERMANENT: 0,      # No expiration
        }
        
        return defaults.get(memory_type, self.settings.DEFAULT_MEMORY_TTL_SECONDS)