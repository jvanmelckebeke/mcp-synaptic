# MCP Synaptic Server - Tool Documentation

## Overview

MCP Synaptic provides **12 MCP tools** across two main categories:
- **Memory Management** (6 tools): Persistent data storage with TTL-based expiration
- **RAG (Document) Operations** (6 tools): Semantic document storage and retrieval

## Memory Tools
*For storing structured data, session state, user preferences, and temporary information*

### `memory_add`
**Purpose**: Store data with automatic expiration and categorization

**Arguments**:
- `key` (string): Unique identifier
- `data` (object): JSON data to store  
- `memory_type` (optional): `"ephemeral"` (5min), `"short_term"` (1hr), `"long_term"` (1week), `"permanent"`
- `ttl_seconds` (optional): Custom expiration time
- `tags` (optional): Key-value categorization
- `metadata` (optional): Additional structured data

**Example**:
```json
{
  "key": "user_session_123",
  "data": {"user_id": 456, "preferences": {"theme": "dark"}},
  "memory_type": "short_term",
  "ttl_seconds": 3600,
  "tags": {"category": "session", "user_type": "premium"}
}
```

### `memory_get`
**Purpose**: Retrieve stored data by key

**Arguments**:
- `key` (string): Memory identifier
- `touch` (boolean, default true): Update access tracking for sliding TTL

### `memory_update`
**Purpose**: Modify existing memory data or extend TTL

**Arguments**:
- `key` (string): Memory identifier
- `data` (optional): New data to replace existing
- `extend_ttl` (optional): New TTL in seconds
- `tags`/`metadata` (optional): Updated categorization

### `memory_delete`
**Purpose**: Remove memory permanently

**Arguments**:
- `key` (string): Memory identifier

### `memory_list`
**Purpose**: Query memories with filtering and pagination

**Arguments**:
- `keys` (optional): Filter to specific keys
- `memory_types` (optional): Filter by type array
- `include_expired` (optional): Include expired entries
- `limit`/`offset` (optional): Pagination controls (limit: 1-100, default 10)

**Example**:
```json
{
  "memory_types": ["short_term", "long_term"],
  "limit": 25,
  "include_expired": false
}
```

### `memory_stats`
**Purpose**: Get storage usage statistics

**Arguments**: None

**Returns**: Total memories, breakdown by type, expired count, size metrics, access patterns

## RAG (Document) Tools
*For storing documents, stories, knowledge base content with semantic search*

### `rag_add_document`
**Purpose**: Store document with automatic embedding generation

**Arguments**:
- `content` (string): Document text content
- `metadata` (optional): Categorization and filtering data
- `document_id` (optional): Custom identifier (auto-generated if omitted)

**Example**:
```json
{
  "content": "Machine learning is a subset of artificial intelligence...",
  "metadata": {
    "category": "technology",
    "author": "John Doe",
    "source": "blog_post"
  },
  "document_id": "ml_intro_001"
}
```

### `rag_get_document`
**Purpose**: Retrieve document by ID

**Arguments**:
- `document_id` (string): Document identifier

### `rag_update_document`
**Purpose**: Update document content or metadata (re-embeds if content changes)

**Arguments**:
- `document_id` (string): Document identifier
- `content` (optional): New document content
- `metadata` (optional): Updated metadata

### `rag_delete_document`
**Purpose**: Remove document and embeddings permanently

**Arguments**:
- `document_id` (string): Document identifier

### `rag_search`
**Purpose**: Semantic similarity search across documents

**Arguments**:
- `query` (string): Search query text
- `limit` (optional): Max results (default 10, max 100)
- `similarity_threshold` (optional): Minimum similarity score (0.0-1.0)
- `metadata_filter` (optional): Filter by metadata key-value pairs

**Example**:
```json
{
  "query": "machine learning algorithms for classification",
  "limit": 5,
  "similarity_threshold": 0.7,
  "metadata_filter": {"category": "technology", "difficulty": "beginner"}
}
```

### `rag_collection_stats`
**Purpose**: Get document collection statistics

**Arguments**: None

**Returns**: Document count, embedding info, size metrics, content analysis

## Key Differences

**Memory Tools** are for:
- Session data, user preferences, application state
- Structured data with TTL-based cleanup
- Fast key-based retrieval
- Temporary caching needs

**RAG Tools** are for:
- Stories, articles, knowledge base content
- Semantic search and similarity matching
- Long-term document storage
- Content that users want to find by meaning, not exact key

## Memory Types & TTL Behavior

- **Ephemeral**: Very short-lived (5 minutes) - for temporary calculations, intermediate results
- **Short-term**: Session-based (1 hour) - for user sessions, current conversation context
- **Long-term**: Persistent (1 week) - for user preferences, important findings
- **Permanent**: Never expires - for critical configuration, learned user patterns

## Configuration Notes

- **Embedding Provider**: Supports both API-based (external services) and local (sentence-transformers) embedding generation
- **Storage Backend**: SQLite (default) or Redis (distributed) storage options
- **Error Handling**: All tools include comprehensive error handling with structured logging
- **Performance**: Memory operations are typically faster than RAG operations due to embedding generation overhead

## Usage Patterns

**For AI Assistants calling this MCP server:**

1. **Use Memory Tools when you need to:**
   - Remember user preferences across conversations
   - Store intermediate calculation results
   - Cache API responses or processed data
   - Track conversation state or context

2. **Use RAG Tools when you need to:**
   - Store user-provided documents, stories, or articles
   - Enable semantic search across stored content
   - Build a knowledge base of information
   - Find relevant content based on meaning rather than exact keywords

3. **Best Practices:**
   - Use appropriate memory types based on how long data should persist
   - Include meaningful metadata for better organization and filtering
   - Use semantic search with appropriate similarity thresholds (0.7+ for precise matches, 0.5+ for broader relevance)
   - Combine metadata filters with semantic search for refined results