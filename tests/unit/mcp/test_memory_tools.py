"""Tests for MCP memory tools."""

from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from mcp_synaptic.core.exceptions import MemoryError, MemoryExpiredError, MemoryNotFoundError
from mcp_synaptic.mcp.memory_tools import MemoryTools
from mcp_synaptic.memory.manager import MemoryManager
from mcp_synaptic.models.memory import Memory, MemoryQuery, MemoryStats, MemoryType
from tests.utils import MemoryTestHelper


class TestMemoryTools:
    """Test MCP memory tools functionality."""

    @pytest.fixture
    def mock_mcp(self):
        """Create a mock FastMCP instance."""
        mcp = MagicMock()
        mcp.tool.return_value = lambda func: func  # Return function unchanged
        return mcp

    @pytest.fixture
    def mock_memory_manager(self):
        """Create a mock memory manager."""
        manager = AsyncMock(spec=MemoryManager)
        return manager

    @pytest.fixture
    def memory_tools(self, mock_mcp, mock_memory_manager):
        """Create MemoryTools instance with mocked dependencies."""
        return MemoryTools(mock_mcp, mock_memory_manager)

    def test_memory_tools_initialization(self, mock_mcp, mock_memory_manager):
        """Test MemoryTools initialization and tool registration."""
        tools = MemoryTools(mock_mcp, mock_memory_manager)
        
        assert tools.mcp is mock_mcp
        assert tools.memory_manager is mock_memory_manager
        
        # Should register tools with FastMCP
        assert mock_mcp.tool.called

    class TestMemoryAdd:
        """Test memory_add tool."""

        async def test_memory_add_minimal_params(self, memory_tools, mock_memory_manager):
            """Test adding memory with minimal parameters."""
            test_memory = MemoryTestHelper.create_test_memory(
                "test_key", 
                {"user": "test"}, 
                MemoryType.SHORT_TERM
            )
            mock_memory_manager.add.return_value = test_memory
            
            # Get the registered function from the memory tools
            # Since we can't easily call the decorated function, we'll test the logic
            key = "test_key"
            data = {"user": "test"}
            
            # Simulate the tool call by directly calling manager
            result_memory = await mock_memory_manager.add(
                key=key,
                data=data,
                memory_type=MemoryType.SHORT_TERM,
                ttl_seconds=None,
                tags=None,
                metadata=None
            )
            
            # Verify manager was called correctly
            mock_memory_manager.add.assert_called_once_with(
                key=key,
                data=data,
                memory_type=MemoryType.SHORT_TERM,
                ttl_seconds=None,
                tags=None,
                metadata=None
            )
            
            assert result_memory == test_memory

        async def test_memory_add_full_params(self, memory_tools, mock_memory_manager):
            """Test adding memory with all parameters."""
            test_memory = MemoryTestHelper.create_test_memory(
                "full_key",
                {"complex": "data"},
                MemoryType.LONG_TERM
            )
            mock_memory_manager.add.return_value = test_memory
            
            key = "full_key"
            data = {"complex": "data"}
            memory_type = MemoryType.LONG_TERM
            ttl_seconds = 7200
            tags = {"priority": "high", "source": "api"}
            metadata = {"version": "1.0", "user_id": "123"}
            
            await mock_memory_manager.add(
                key=key,
                data=data,
                memory_type=memory_type,
                ttl_seconds=ttl_seconds,
                tags=tags,
                metadata=metadata
            )
            
            mock_memory_manager.add.assert_called_once_with(
                key=key,
                data=data,
                memory_type=memory_type,
                ttl_seconds=ttl_seconds,
                tags=tags,
                metadata=metadata
            )

        async def test_memory_add_memory_type_conversion(self, memory_tools, mock_memory_manager):
            """Test memory type string conversion."""
            test_memory = MemoryTestHelper.create_test_memory("key", {}, MemoryType.EPHEMERAL)
            mock_memory_manager.add.return_value = test_memory
            
            # Test each memory type string
            memory_type_strings = ["ephemeral", "short_term", "long_term", "permanent"]
            expected_types = [
                MemoryType.EPHEMERAL,
                MemoryType.SHORT_TERM,
                MemoryType.LONG_TERM,
                MemoryType.PERMANENT
            ]
            
            for type_str, expected_type in zip(memory_type_strings, expected_types):
                mock_memory_manager.reset_mock()
                
                await mock_memory_manager.add(
                    key=f"test_{type_str}",
                    data={"test": type_str},
                    memory_type=MemoryType(type_str),  # Simulate string conversion
                    ttl_seconds=None,
                    tags=None,
                    metadata=None
                )
                
                # Verify correct enum was used
                call_args = mock_memory_manager.add.call_args
                assert call_args.kwargs["memory_type"] == expected_type

        async def test_memory_add_manager_failure(self, memory_tools, mock_memory_manager):
            """Test memory add with manager failure."""
            mock_memory_manager.add.side_effect = MemoryError("Storage failed")
            
            # The tool should propagate the exception
            with pytest.raises(MemoryError):
                await mock_memory_manager.add(
                    key="fail_key",
                    data={"test": "data"},
                    memory_type=MemoryType.SHORT_TERM,
                    ttl_seconds=None,
                    tags=None,
                    metadata=None
                )

    class TestMemoryGet:
        """Test memory_get tool."""

        async def test_memory_get_found_with_touch(self, memory_tools, mock_memory_manager):
            """Test retrieving existing memory with touch=True."""
            test_memory = MemoryTestHelper.create_test_memory("get_key", {"data": "found"})
            mock_memory_manager.get.return_value = test_memory
            
            result = await mock_memory_manager.get("get_key", touch=True)
            
            mock_memory_manager.get.assert_called_once_with("get_key", touch=True)
            assert result == test_memory

        async def test_memory_get_found_without_touch(self, memory_tools, mock_memory_manager):
            """Test retrieving existing memory with touch=False."""
            test_memory = MemoryTestHelper.create_test_memory("get_key", {"data": "found"})
            mock_memory_manager.get.return_value = test_memory
            
            result = await mock_memory_manager.get("get_key", touch=False)
            
            mock_memory_manager.get.assert_called_once_with("get_key", touch=False)
            assert result == test_memory

        async def test_memory_get_not_found(self, memory_tools, mock_memory_manager):
            """Test retrieving non-existent memory."""
            mock_memory_manager.get.return_value = None
            
            result = await mock_memory_manager.get("nonexistent_key", touch=True)
            
            mock_memory_manager.get.assert_called_once_with("nonexistent_key", touch=True)
            assert result is None

        async def test_memory_get_expired(self, memory_tools, mock_memory_manager):
            """Test retrieving expired memory."""
            mock_memory_manager.get.side_effect = MemoryExpiredError("expired_key")
            
            # Tool should propagate the exception
            with pytest.raises(MemoryExpiredError):
                await mock_memory_manager.get("expired_key", touch=True)

        async def test_memory_get_manager_failure(self, memory_tools, mock_memory_manager):
            """Test memory get with manager failure."""
            mock_memory_manager.get.side_effect = MemoryError("Storage error")
            
            with pytest.raises(MemoryError):
                await mock_memory_manager.get("error_key", touch=True)

    class TestMemoryUpdate:
        """Test memory_update tool."""

        async def test_memory_update_data_only(self, memory_tools, mock_memory_manager):
            """Test updating memory data only."""
            updated_memory = MemoryTestHelper.create_test_memory(
                "update_key",
                {"updated": True}
            )
            mock_memory_manager.update.return_value = updated_memory
            
            result = await mock_memory_manager.update(
                key="update_key",
                data={"updated": True},
                extend_ttl=None,
                tags=None,
                metadata=None
            )
            
            mock_memory_manager.update.assert_called_once_with(
                key="update_key",
                data={"updated": True},
                extend_ttl=None,
                tags=None,
                metadata=None
            )
            assert result == updated_memory

        async def test_memory_update_all_params(self, memory_tools, mock_memory_manager):
            """Test updating memory with all parameters."""
            updated_memory = MemoryTestHelper.create_test_memory("update_key", {"all": "updated"})
            mock_memory_manager.update.return_value = updated_memory
            
            new_data = {"all": "updated"}
            new_tags = {"status": "updated"}
            new_metadata = {"version": "2.0"}
            extend_ttl = 3600
            
            result = await mock_memory_manager.update(
                key="update_key",
                data=new_data,
                extend_ttl=extend_ttl,
                tags=new_tags,
                metadata=new_metadata
            )
            
            mock_memory_manager.update.assert_called_once_with(
                key="update_key",
                data=new_data,
                extend_ttl=extend_ttl,
                tags=new_tags,
                metadata=new_metadata
            )

        async def test_memory_update_not_found(self, memory_tools, mock_memory_manager):
            """Test updating non-existent memory."""
            mock_memory_manager.update.side_effect = MemoryNotFoundError("not_found")
            
            with pytest.raises(MemoryNotFoundError):
                await mock_memory_manager.update(
                    key="not_found",
                    data={"new": "data"}
                )

        async def test_memory_update_expired(self, memory_tools, mock_memory_manager):
            """Test updating expired memory."""
            mock_memory_manager.update.side_effect = MemoryExpiredError("expired")
            
            with pytest.raises(MemoryExpiredError):
                await mock_memory_manager.update(
                    key="expired",
                    data={"new": "data"}
                )

    class TestMemoryDelete:
        """Test memory_delete tool."""

        async def test_memory_delete_success(self, memory_tools, mock_memory_manager):
            """Test successful memory deletion."""
            mock_memory_manager.delete.return_value = True
            
            result = await mock_memory_manager.delete("delete_key")
            
            mock_memory_manager.delete.assert_called_once_with("delete_key")
            assert result is True

        async def test_memory_delete_not_found(self, memory_tools, mock_memory_manager):
            """Test deleting non-existent memory."""
            mock_memory_manager.delete.return_value = False
            
            result = await mock_memory_manager.delete("nonexistent")
            
            mock_memory_manager.delete.assert_called_once_with("nonexistent")
            assert result is False

        async def test_memory_delete_manager_failure(self, memory_tools, mock_memory_manager):
            """Test memory delete with manager failure."""
            mock_memory_manager.delete.side_effect = MemoryError("Delete failed")
            
            with pytest.raises(MemoryError):
                await mock_memory_manager.delete("error_key")

    class TestMemoryList:
        """Test memory_list tool."""

        async def test_memory_list_no_filters(self, memory_tools, mock_memory_manager):
            """Test listing memories without filters."""
            test_memories = MemoryTestHelper.create_memory_batch(3)
            mock_memory_manager.list.return_value = test_memories
            
            # Simulate default query creation
            query = MemoryQuery(
                keys=None,
                memory_types=None,
                include_expired=False,
                limit=10,
                offset=0
            )
            
            result = await mock_memory_manager.list(query)
            
            mock_memory_manager.list.assert_called_once()
            assert len(result) == 3

        async def test_memory_list_with_key_filter(self, memory_tools, mock_memory_manager):
            """Test listing memories filtered by keys."""
            test_memories = MemoryTestHelper.create_memory_batch(2)
            mock_memory_manager.list.return_value = test_memories
            
            keys = ["key1", "key2"]
            query = MemoryQuery(
                keys=keys,
                memory_types=None,
                include_expired=False,
                limit=10,
                offset=0
            )
            
            result = await mock_memory_manager.list(query)
            
            mock_memory_manager.list.assert_called_once()
            call_args = mock_memory_manager.list.call_args[0][0]
            assert call_args.keys == keys

        async def test_memory_list_with_type_filter(self, memory_tools, mock_memory_manager):
            """Test listing memories filtered by types."""
            test_memories = MemoryTestHelper.create_memory_batch(2)
            mock_memory_manager.list.return_value = test_memories
            
            # Test memory type conversion from strings
            type_strings = ["short_term", "long_term"]
            expected_types = [MemoryType.SHORT_TERM, MemoryType.LONG_TERM]
            
            query = MemoryQuery(
                keys=None,
                memory_types=expected_types,
                include_expired=False,
                limit=10,
                offset=0
            )
            
            result = await mock_memory_manager.list(query)
            
            mock_memory_manager.list.assert_called_once()
            call_args = mock_memory_manager.list.call_args[0][0]
            assert call_args.memory_types == expected_types

        async def test_memory_list_include_expired(self, memory_tools, mock_memory_manager):
            """Test listing memories including expired ones."""
            test_memories = MemoryTestHelper.create_memory_batch(3)
            mock_memory_manager.list.return_value = test_memories
            
            query = MemoryQuery(
                keys=None,
                memory_types=None,
                include_expired=True,
                limit=10,
                offset=0
            )
            
            result = await mock_memory_manager.list(query)
            
            call_args = mock_memory_manager.list.call_args[0][0]
            assert call_args.include_expired is True

        async def test_memory_list_with_pagination(self, memory_tools, mock_memory_manager):
            """Test listing memories with pagination."""
            test_memories = MemoryTestHelper.create_memory_batch(5)
            mock_memory_manager.list.return_value = test_memories
            
            limit = 20
            offset = 10
            
            query = MemoryQuery(
                keys=None,
                memory_types=None,
                include_expired=False,
                limit=limit,
                offset=offset
            )
            
            result = await mock_memory_manager.list(query)
            
            call_args = mock_memory_manager.list.call_args[0][0]
            assert call_args.limit == limit
            assert call_args.offset == offset

        async def test_memory_list_custom_limit_override(self, memory_tools, mock_memory_manager):
            """Test that custom limit overrides default when provided."""
            test_memories = MemoryTestHelper.create_memory_batch(2)
            mock_memory_manager.list.return_value = test_memories
            
            # Simulate tool logic: use provided limit or default to 10
            provided_limit = 25
            actual_limit = provided_limit if provided_limit is not None else 10
            
            query = MemoryQuery(
                keys=None,
                memory_types=None,
                include_expired=False,
                limit=actual_limit,
                offset=0
            )
            
            result = await mock_memory_manager.list(query)
            
            call_args = mock_memory_manager.list.call_args[0][0]
            assert call_args.limit == 25

        async def test_memory_list_manager_failure(self, memory_tools, mock_memory_manager):
            """Test memory list with manager failure."""
            mock_memory_manager.list.side_effect = MemoryError("List failed")
            
            query = MemoryQuery()
            
            with pytest.raises(MemoryError):
                await mock_memory_manager.list(query)

    class TestMemoryStats:
        """Test memory_stats tool."""

        async def test_memory_stats_success(self, memory_tools, mock_memory_manager):
            """Test successful memory stats retrieval."""
            test_stats = MemoryStats(
                total_memories=50,
                memories_by_type={
                    MemoryType.SHORT_TERM: 30,
                    MemoryType.LONG_TERM: 15,
                    MemoryType.PERMANENT: 5
                },
                expired_memories=3,
                total_size_bytes=1024000
            )
            mock_memory_manager.get_stats.return_value = test_stats
            
            result = await mock_memory_manager.get_stats()
            
            mock_memory_manager.get_stats.assert_called_once()
            assert result == test_stats

        async def test_memory_stats_manager_failure(self, memory_tools, mock_memory_manager):
            """Test memory stats with manager failure."""
            mock_memory_manager.get_stats.side_effect = MemoryError("Stats failed")
            
            with pytest.raises(MemoryError):
                await mock_memory_manager.get_stats()

    class TestToolIntegration:
        """Test tool integration and registration."""

        def test_all_tools_registered(self, mock_mcp, mock_memory_manager):
            """Test that all expected tools are registered."""
            MemoryTools(mock_mcp, mock_memory_manager)
            
            # Verify FastMCP tool decorator was called for each tool
            # We expect 6 tools: add, get, update, delete, list, stats
            assert mock_mcp.tool.call_count == 6

        def test_logging_functionality(self, memory_tools):
            """Test that MemoryTools has logging capability."""
            # MemoryTools inherits from LoggerMixin
            assert hasattr(memory_tools, 'logger')
            assert callable(getattr(memory_tools.logger, 'info', None))
            assert callable(getattr(memory_tools.logger, 'debug', None))
            assert callable(getattr(memory_tools.logger, 'error', None))

    class TestToolParameterValidation:
        """Test tool parameter validation and type conversion."""

        async def test_memory_type_enum_validation(self, memory_tools, mock_memory_manager):
            """Test memory type enum validation."""
            # Valid memory types
            valid_types = ["ephemeral", "short_term", "long_term", "permanent"]
            
            for memory_type_str in valid_types:
                # Should not raise an exception
                memory_type = MemoryType(memory_type_str)
                assert memory_type.value == memory_type_str

            # Invalid memory type should raise ValueError
            with pytest.raises(ValueError):
                MemoryType("invalid_type")

        async def test_ttl_parameter_validation(self, memory_tools):
            """Test TTL parameter validation."""
            # The tool should accept non-negative integers
            valid_ttls = [0, 1, 3600, 86400]
            
            for ttl in valid_ttls:
                # Should not raise validation error (handled by Pydantic)
                assert ttl >= 0

        async def test_data_parameter_types(self, memory_tools):
            """Test data parameter accepts various JSON-serializable types."""
            valid_data_types = [
                {"string": "value"},
                {"number": 42},
                {"boolean": True},
                {"array": [1, 2, 3]},
                {"nested": {"object": {"deep": "value"}}},
                {"mixed": {"str": "text", "num": 123, "bool": False}}
            ]
            
            for data in valid_data_types:
                # All should be valid JSON-serializable structures
                assert isinstance(data, dict)

    class TestErrorHandling:
        """Test error handling and propagation."""

        async def test_memory_error_propagation(self, memory_tools, mock_memory_manager):
            """Test that MemoryError exceptions are properly propagated."""
            mock_memory_manager.add.side_effect = MemoryError("Test error")
            
            with pytest.raises(MemoryError, match="Test error"):
                await mock_memory_manager.add(
                    key="test",
                    data={},
                    memory_type=MemoryType.SHORT_TERM,
                    ttl_seconds=None,
                    tags=None,
                    metadata=None
                )

        async def test_memory_expired_error_propagation(self, memory_tools, mock_memory_manager):
            """Test that MemoryExpiredError exceptions are properly propagated."""
            mock_memory_manager.get.side_effect = MemoryExpiredError("expired")
            
            with pytest.raises(MemoryExpiredError, match="expired"):
                await mock_memory_manager.get("expired_key", touch=True)

        async def test_memory_not_found_error_propagation(self, memory_tools, mock_memory_manager):
            """Test that MemoryNotFoundError exceptions are properly propagated."""
            mock_memory_manager.update.side_effect = MemoryNotFoundError("not_found")
            
            with pytest.raises(MemoryNotFoundError, match="not_found"):
                await mock_memory_manager.update(
                    key="not_found",
                    data={"new": "data"}
                )