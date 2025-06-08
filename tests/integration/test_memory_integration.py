"""Integration tests for memory system after refactoring."""

import asyncio
import pytest
from pathlib import Path
import tempfile
from datetime import datetime, timedelta, UTC

from mcp_synaptic.config.settings import Settings
from mcp_synaptic.memory.manager import MemoryManager
from mcp_synaptic.models.memory import Memory, MemoryType, MemoryQuery


@pytest.fixture
def integration_settings():
    """Settings for integration testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        settings = Settings(
            SQLITE_DATABASE_PATH=Path(temp_dir) / "test_memory.db",
            REDIS_ENABLED=False,
            DEFAULT_MEMORY_TTL_SECONDS=3600
        )
        yield settings


class TestMemoryManagerIntegration:
    """Test memory manager integration with real storage backends."""

    @pytest.mark.asyncio
    async def test_memory_lifecycle_sqlite_integration(self, integration_settings):
        """Test complete memory lifecycle with SQLite storage."""
        # Initialize memory manager with real SQLite storage
        manager = MemoryManager(integration_settings)
        await manager.initialize()
        
        try:
            # Test add memory
            key = "integration_test_key"
            data = {"test": "data", "number": 42}
            memory_type = MemoryType.SHORT_TERM
            
            added_memory = await manager.add(
                key=key,
                data=data,
                memory_type=memory_type,
                ttl_seconds=7200
            )
            
            assert added_memory.key == key
            assert added_memory.data == data
            assert added_memory.memory_type == memory_type
            assert added_memory.ttl_seconds == 7200
            
            # Test get memory
            retrieved_memory = await manager.get(key)
            assert retrieved_memory is not None
            assert retrieved_memory.key == key
            assert retrieved_memory.data == data
            
            # Test update memory
            new_data = {"updated": "data", "number": 84}
            updated_memory = await manager.update(key, data=new_data)
            assert updated_memory.data == new_data
            
            # Test list memories
            memories = await manager.list()
            assert len(memories) >= 1
            assert any(m.key == key for m in memories)
            
            # Test exists
            exists = await manager.exists(key)
            assert exists is True
            
            # Test touch
            touched = await manager.touch(key)
            assert touched is True
            
            # Test stats
            stats = await manager.get_stats()
            assert stats.total_memories >= 1
            
            # Test delete memory
            deleted = await manager.delete(key)
            assert deleted is True
            
            # Verify deletion
            retrieved_after_delete = await manager.get(key)
            assert retrieved_after_delete is None
            
        finally:
            await manager.close()

    @pytest.mark.asyncio
    async def test_memory_expiration_integration(self, integration_settings):
        """Test memory expiration with real storage."""
        manager = MemoryManager(integration_settings)
        await manager.initialize()
        
        try:
            # Add memory with short TTL
            key = "expiring_memory"
            data = {"will": "expire"}
            ttl_seconds = 1  # 1 second
            
            await manager.add(
                key=key,
                data=data,
                ttl_seconds=ttl_seconds
            )
            
            # Verify memory exists
            memory = await manager.get(key)
            assert memory is not None
            
            # Wait for expiration
            await asyncio.sleep(1.5)
            
            # Clean up expired memories
            cleaned_count = await manager.cleanup_expired()
            assert cleaned_count >= 1
            
            # Verify memory is gone
            expired_memory = await manager.get(key)
            assert expired_memory is None
            
        finally:
            await manager.close()

    @pytest.mark.asyncio
    async def test_memory_query_filtering_integration(self, integration_settings):
        """Test memory query filtering with real storage."""
        manager = MemoryManager(integration_settings)
        await manager.initialize()
        
        try:
            # Add multiple memories with different types
            await manager.add("ephemeral_key", {"type": "ephemeral"}, MemoryType.EPHEMERAL)
            await manager.add("short_key", {"type": "short"}, MemoryType.SHORT_TERM)
            await manager.add("long_key", {"type": "long"}, MemoryType.LONG_TERM)
            
            # Test filtering by memory type
            ephemeral_query = MemoryQuery(memory_types=[MemoryType.EPHEMERAL])
            ephemeral_memories = await manager.list(ephemeral_query)
            assert len(ephemeral_memories) >= 1
            assert all(m.memory_type == MemoryType.EPHEMERAL for m in ephemeral_memories)
            
            # Test limiting results
            limited_query = MemoryQuery(limit=2)
            limited_memories = await manager.list(limited_query)
            assert len(limited_memories) <= 2
            
            # Test specific key filtering
            key_query = MemoryQuery(keys=["short_key"])
            key_memories = await manager.list(key_query)
            assert len(key_memories) >= 1
            assert all(m.key == "short_key" for m in key_memories)
            
        finally:
            await manager.close()

    @pytest.mark.asyncio
    async def test_concurrent_memory_operations_integration(self, integration_settings):
        """Test concurrent memory operations with real storage."""
        manager = MemoryManager(integration_settings)
        await manager.initialize()
        
        try:
            # Prepare concurrent operations
            keys = [f"concurrent_key_{i}" for i in range(10)]
            
            # Add memories concurrently
            add_tasks = [
                manager.add(key, {"index": i, "data": f"value_{i}"})
                for i, key in enumerate(keys)
            ]
            added_memories = await asyncio.gather(*add_tasks)
            assert len(added_memories) == 10
            
            # Get memories concurrently
            get_tasks = [manager.get(key) for key in keys]
            retrieved_memories = await asyncio.gather(*get_tasks)
            assert len(retrieved_memories) == 10
            assert all(m is not None for m in retrieved_memories)
            
            # Update memories concurrently
            update_tasks = [
                manager.update(key, data={"updated": True, "index": i})
                for i, key in enumerate(keys)
            ]
            updated_memories = await asyncio.gather(*update_tasks)
            assert len(updated_memories) == 10
            assert all(m.data["updated"] is True for m in updated_memories)
            
            # Delete memories concurrently
            delete_tasks = [manager.delete(key) for key in keys]
            delete_results = await asyncio.gather(*delete_tasks)
            assert all(deleted is True for deleted in delete_results)
            
        finally:
            await manager.close()

    @pytest.mark.asyncio
    async def test_memory_error_handling_integration(self, integration_settings):
        """Test error handling in memory operations with real storage."""
        manager = MemoryManager(integration_settings)
        await manager.initialize()
        
        try:
            # Test operations on non-existent memory
            non_existent_memory = await manager.get("does_not_exist")
            assert non_existent_memory is None
            
            exists_result = await manager.exists("does_not_exist")
            assert exists_result is False
            
            touch_result = await manager.touch("does_not_exist")
            assert touch_result is False
            
            delete_result = await manager.delete("does_not_exist")
            assert delete_result is False
            
            # Test update on non-existent memory
            from mcp_synaptic.core.exceptions import MemoryNotFoundError
            with pytest.raises(MemoryNotFoundError):
                await manager.update("does_not_exist", data={"new": "data"})
            
        finally:
            await manager.close()


class TestMemoryStorageBackendIntegration:
    """Test memory manager with different storage backends."""

    @pytest.mark.asyncio
    async def test_sqlite_storage_backend_integration(self, integration_settings):
        """Test memory manager specifically with SQLite storage."""
        integration_settings.REDIS_ENABLED = False
        
        manager = MemoryManager(integration_settings)
        await manager.initialize()
        
        try:
            # Verify SQLite storage is used
            from mcp_synaptic.memory.storage.sqlite import SQLiteMemoryStorage
            assert isinstance(manager.storage, SQLiteMemoryStorage)
            
            # Test basic operations work with SQLite
            memory = await manager.add("sqlite_test", {"backend": "sqlite"})
            assert memory.key == "sqlite_test"
            
            retrieved = await manager.get("sqlite_test")
            assert retrieved.data["backend"] == "sqlite"
            
        finally:
            await manager.close()

    @pytest.mark.asyncio
    async def test_memory_persistence_across_sessions(self, integration_settings):
        """Test that memory persists across manager sessions."""
        key = "persistent_memory"
        data = {"should": "persist"}
        
        # First session - add memory
        manager1 = MemoryManager(integration_settings)
        await manager1.initialize()
        
        try:
            await manager1.add(key, data, MemoryType.PERMANENT)
        finally:
            await manager1.close()
        
        # Second session - retrieve memory
        manager2 = MemoryManager(integration_settings)
        await manager2.initialize()
        
        try:
            retrieved = await manager2.get(key)
            assert retrieved is not None
            assert retrieved.data == data
            assert retrieved.memory_type == MemoryType.PERMANENT
        finally:
            await manager2.close()