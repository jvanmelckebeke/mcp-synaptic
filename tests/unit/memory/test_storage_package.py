"""Tests for the refactored memory storage package structure."""

import pytest

from mcp_synaptic.config.settings import Settings
from mcp_synaptic.core.exceptions import DatabaseError, ConnectionError


class TestStoragePackageStructure:
    """Test the new memory storage package organization."""
    
    def test_base_module_imports(self):
        """Test that base storage interface can be imported."""
        from mcp_synaptic.memory.storage.base import MemoryStorage
        
        # Test that it's abstract
        assert hasattr(MemoryStorage, '__abstractmethods__')
        assert len(MemoryStorage.__abstractmethods__) > 0
        
        # Test that we can't instantiate it directly
        with pytest.raises(TypeError):
            MemoryStorage()
    
    def test_sqlite_module_imports(self):
        """Test that SQLite storage can be imported and instantiated."""
        from mcp_synaptic.memory.storage.sqlite import SQLiteMemoryStorage
        
        settings = Settings()
        storage = SQLiteMemoryStorage(settings)
        
        assert storage.settings == settings
        assert storage.db_path == settings.SQLITE_DATABASE_PATH
        assert storage._connection is None
    
    def test_redis_module_imports(self):
        """Test that Redis storage can be imported and instantiated."""
        from mcp_synaptic.memory.storage.redis import RedisMemoryStorage
        
        settings = Settings()
        storage = RedisMemoryStorage(settings)
        
        assert storage.settings == settings
        assert storage.redis_url == settings.REDIS_URL
        assert storage._redis is None
    
    def test_storage_inheritance(self):
        """Test that concrete implementations inherit from base."""
        from mcp_synaptic.memory.storage.base import MemoryStorage
        from mcp_synaptic.memory.storage.sqlite import SQLiteMemoryStorage
        from mcp_synaptic.memory.storage.redis import RedisMemoryStorage
        
        assert issubclass(SQLiteMemoryStorage, MemoryStorage)
        assert issubclass(RedisMemoryStorage, MemoryStorage)
    
    def test_memory_package_exports(self):
        """Test that memory package still exports storage correctly."""
        from mcp_synaptic.memory import MemoryStorage
        from mcp_synaptic.memory.storage.base import MemoryStorage as BaseStorage
        
        # Should be the same class
        assert MemoryStorage is BaseStorage
    
    def test_manager_imports_work(self):
        """Test that manager can import from new structure."""
        from mcp_synaptic.memory.manager import MemoryManager
        
        settings = Settings()
        manager = MemoryManager(settings)
        
        # Should be able to create without import errors
        assert manager.settings == settings
        assert manager.storage is None
        assert not manager._initialized
    
    def test_storage_interface_completeness(self):
        """Test that all required methods are defined in the interface."""
        from mcp_synaptic.memory.storage.base import MemoryStorage
        
        required_methods = {
            'initialize', 'close', 'store', 'retrieve', 
            'delete', 'list_memories', 'cleanup_expired', 'get_stats'
        }
        
        abstract_methods = set(MemoryStorage.__abstractmethods__)
        assert required_methods == abstract_methods
    
    def test_sqlite_implements_all_methods(self):
        """Test that SQLite implementation has all required methods."""
        from mcp_synaptic.memory.storage.sqlite import SQLiteMemoryStorage
        
        required_methods = {
            'initialize', 'close', 'store', 'retrieve', 
            'delete', 'list_memories', 'cleanup_expired', 'get_stats'
        }
        
        sqlite_methods = set(dir(SQLiteMemoryStorage))
        assert required_methods.issubset(sqlite_methods)
    
    def test_redis_implements_all_methods(self):
        """Test that Redis implementation has all required methods."""
        from mcp_synaptic.memory.storage.redis import RedisMemoryStorage
        
        required_methods = {
            'initialize', 'close', 'store', 'retrieve', 
            'delete', 'list_memories', 'cleanup_expired', 'get_stats'
        }
        
        redis_methods = set(dir(RedisMemoryStorage))
        assert required_methods.issubset(redis_methods)
    
    def test_storage_package_structure(self):
        """Test that storage package is properly structured."""
        import mcp_synaptic.memory.storage as storage_package
        
        # Package should have a docstring
        assert storage_package.__doc__ is not None
        assert "storage implementations" in storage_package.__doc__.lower()
    
    def test_domain_separation_is_clean(self):
        """Test that each storage implementation is properly isolated."""
        # Base interface should contain the MemoryStorage class
        from mcp_synaptic.memory.storage import base
        
        # Base should have the MemoryStorage class
        assert hasattr(base, 'MemoryStorage')
        assert hasattr(base.MemoryStorage, '__abstractmethods__')
        
        # SQLite should not depend on Redis
        from mcp_synaptic.memory.storage import sqlite
        sqlite_source = sqlite.__file__
        with open(sqlite_source, 'r') as f:
            sqlite_content = f.read()
        assert 'redis' not in sqlite_content.lower()
        
        # Redis should not depend on SQLite specifics
        from mcp_synaptic.memory.storage import redis
        redis_source = redis.__file__
        with open(redis_source, 'r') as f:
            redis_content = f.read()
        assert 'sqlite' not in redis_content.lower()
    
    def test_error_handling_imports(self):
        """Test that storage modules import correct exceptions."""
        from mcp_synaptic.memory.storage.sqlite import SQLiteMemoryStorage
        from mcp_synaptic.memory.storage.redis import RedisMemoryStorage
        
        # Both should be able to raise appropriate exceptions
        settings = Settings()
        
        sqlite_storage = SQLiteMemoryStorage(settings)
        redis_storage = RedisMemoryStorage(settings)
        
        # Test that exception types are available (they should import them)
        assert hasattr(sqlite_storage, 'logger')  # From LoggerMixin
        assert hasattr(redis_storage, 'logger')   # From LoggerMixin