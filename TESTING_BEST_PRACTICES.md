# Python Testing Best Practices & Anti-Patterns

## Summary
This document outlines critical testing best practices for MCP Synaptic, focusing on proper mocking, test isolation, and pytest usage.

## 🟢 BEST PRACTICES TO FOLLOW

### 1. **Always Use `autospec=True` for Mocking**
```python
# ✅ GOOD: Auto-spec ensures mock respects real object's API
with patch('module.ClassName', autospec=True) as mock_class:
    mock_instance = mock_class.return_value
    
# ❌ BAD: No autospec allows invalid method calls
with patch('module.ClassName') as mock_class:
    mock_class.invalid_method()  # Won't catch this error
```

### 2. **Mock at the Import Location, Not Definition Location**
```python
# ✅ GOOD: Patch where object is imported/used
# If mymodule.py does: from external import ApiClient
@patch('mymodule.ApiClient')  # Patch where it's used

# ❌ BAD: Patching where it's defined
@patch('external.ApiClient')  # Wrong location
```

### 3. **Use Dependency Injection for Testability**
```python
# ✅ GOOD: Injectable dependencies
class UserService:
    def __init__(self, db_client, api_client):
        self.db = db_client
        self.api = api_client

# ❌ BAD: Hard-coded dependencies
class UserService:
    def __init__(self):
        self.db = DatabaseClient()  # Hard to mock
        self.api = ApiClient()      # Hard to mock
```

### 4. **Follow AAA Pattern (Arrange, Act, Assert)**
```python
async def test_user_creation():
    # Arrange
    user_data = {"name": "John", "email": "john@test.com"}
    mock_db = AsyncMock()
    service = UserService(mock_db)
    
    # Act
    result = await service.create_user(user_data)
    
    # Assert
    assert result.name == "John"
    mock_db.save.assert_called_once_with(user_data)
```

### 5. **Use Proper Test Isolation**
```python
# ✅ GOOD: Clean fixtures with proper teardown
@pytest_asyncio.fixture
async def clean_storage():
    storage = await create_test_storage()
    yield storage
    await storage.close()
    # Clean up any files/connections
```

## 🔴 ANTI-PATTERNS TO AVOID

### 1. **Over-Mocking (Implementation Coupling)**
```python
# ❌ BAD: Testing implementation details
def test_user_service():
    with patch.object(service, '_validate_email') as mock_validate:
        with patch.object(service, '_hash_password') as mock_hash:
            service.create_user(data)
            mock_validate.assert_called_once()  # Testing internal methods
            mock_hash.assert_called_once()      # Coupled to implementation
```

### 2. **Not Resetting Mocks Between Tests**
```python
# ❌ BAD: Mock state bleeds between tests
class TestClass:
    @patch('module.external_api')
    def test_first(self, mock_api):
        mock_api.call_count = 5  # Side effect
        
    def test_second(self, mock_api):
        # mock_api still has call_count=5 from previous test!
```

### 3. **Testing Against Real External Resources**
```python
# ❌ BAD: Real database/API calls in unit tests
def test_user_creation():
    db = PostgresConnection("real_db_url")  # Real DB!
    api = HTTPClient("https://real-api.com")  # Real API!
    result = create_user(db, api, user_data)
```

### 4. **Mixed Unit/Integration Concerns**
```python
# ❌ BAD: Unit test that's actually integration test
def test_memory_manager_add():
    # This creates real SQLite file and tests entire stack
    manager = MemoryManager(real_settings)
    await manager.initialize()  # Real DB connection
    result = await manager.add_memory(data)  # Tests storage + manager
```

## 📋 AUDIT CHECKLIST

### Unit Tests Should:
- [ ] Mock all external dependencies (DB, APIs, file system)
- [ ] Use `autospec=True` for all mocks
- [ ] Test single units of behavior
- [ ] Be fast (< 100ms each)
- [ ] Be isolated (no shared state)
- [ ] Use descriptive test names

### Integration Tests Should:
- [ ] Test component interactions
- [ ] Use test databases/containers
- [ ] Clean up resources properly
- [ ] Be in separate test directories
- [ ] Run less frequently than unit tests

### Async Tests Should:
- [ ] Use `pytest_asyncio.fixture` for async fixtures
- [ ] Properly await all async operations
- [ ] Mock async dependencies with `AsyncMock`
- [ ] Handle async context managers correctly

## 🚨 CURRENT ISSUES IDENTIFIED

1. **File System Dependencies**: Tests creating real database files
2. **Missing Autospec**: Mocks without proper API enforcement
3. **Test Interdependence**: Database state bleeding between tests
4. **Mixed Concerns**: Unit tests doing integration work
5. **Improper Async Mocking**: Redis async iterator issues

## NEXT STEPS

1. **Audit Current Tests**: Map unit vs integration boundaries
2. **Redesign Architecture**: Separate concerns properly
3. **Implement Proper Mocking**: Add autospec and dependency injection
4. **Create Test Categories**: Unit, integration, and E2E test separation
5. **Setup Test Infrastructure**: Proper fixtures and cleanup