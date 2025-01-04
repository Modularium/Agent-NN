# Advanced API Endpoints

This document describes the advanced API endpoints for model management, knowledge base operations, and system administration.

## Model Management

### Create Model

```http
POST /models
```

Create a new model configuration.

Request body:
```json
{
    "name": "string",
    "type": "string (llm, nn, hybrid)",
    "source": "string (local, huggingface, openai)",
    "config": {
        "key": "value"
    },
    "version": "string (optional)"
}
```

Response:
```json
{
    "name": "string",
    "type": "string",
    "source": "string",
    "config": {
        "key": "value"
    },
    "version": "string"
}
```

### List Model Versions

```http
GET /models/{model_name}/versions
```

List available versions of a model.

Response:
```json
[
    "v1",
    "v2",
    "v3"
]
```

## Knowledge Base Management

### Create Knowledge Base

```http
POST /knowledge-bases
```

Create a new knowledge base.

Request body:
```json
{
    "name": "string",
    "domain": "string",
    "sources": ["string"],
    "update_interval": "integer"
}
```

Response:
```json
{
    "name": "string",
    "domain": "string",
    "sources": ["string"],
    "update_interval": "integer"
}
```

### Upload Documents

```http
POST /knowledge-bases/{kb_name}/documents
```

Upload documents to a knowledge base.

Request body:
- Form data with files

Response:
```json
{
    "uploaded": [
        {
            "filename": "string",
            "doc_id": "string"
        }
    ]
}
```

## System Administration

### Update System Configuration

```http
POST /system/config
```

Update system configuration.

Request body:
```json
{
    "max_concurrent_tasks": "integer",
    "task_timeout": "integer",
    "cache_size": "integer",
    "log_level": "string"
}
```

Response:
```json
{
    "max_concurrent_tasks": "integer",
    "task_timeout": "integer",
    "cache_size": "integer",
    "log_level": "string"
}
```

### Create Backup

```http
POST /system/backup
```

Create system backup.

Request body:
```json
{
    "target_dir": "string",
    "include_models": "boolean",
    "include_data": "boolean",
    "max_backups": "integer"
}
```

Response:
```json
{
    "target_dir": "string",
    "include_models": "boolean",
    "include_data": "boolean",
    "max_backups": "integer",
    "backup_id": "string",
    "timestamp": "string",
    "size": "integer"
}
```

### Restore Backup

```http
POST /system/restore/{backup_id}
```

Restore system from backup.

Response:
```json
{
    "status": "restored",
    "backup_id": "string"
}
```

## Cache Management

### Clear Cache

```http
POST /system/cache/clear
```

Clear system cache.

Response:
```json
{
    "status": "cleared"
}
```

### Get Cache Statistics

```http
GET /system/cache/stats
```

Get cache statistics.

Response:
```json
{
    "size": "integer",
    "items": "integer",
    "hit_rate": "float",
    "miss_rate": "float",
    "evictions": "integer"
}
```

## Error Responses

All endpoints may return the following error responses:

### 401 Unauthorized
```json
{
    "detail": "Invalid credentials"
}
```

### 404 Not Found
```json
{
    "detail": "Resource not found"
}
```

### 500 Internal Server Error
```json
{
    "detail": "Error message"
}
```

## Implementation Notes

### Model Sources

1. Local Models:
   - Models stored in local filesystem
   - Custom trained models
   - Requires model file path in config

2. HuggingFace Models:
   - Models from HuggingFace Hub
   - Requires model ID and optional revision
   - Supports automatic downloading

3. OpenAI Models:
   - OpenAI API models
   - Requires API key and model name
   - Supports configuration options

### Knowledge Base Features

1. Document Processing:
   - Automatic text extraction
   - Metadata extraction
   - Content validation
   - Duplicate detection

2. Data Sources:
   - Local files
   - URLs
   - Databases
   - APIs

3. Update Intervals:
   - Scheduled updates
   - Manual updates
   - Real-time updates

### System Administration

1. Configuration:
   - Runtime configuration
   - Persistent configuration
   - Environment overrides

2. Backup:
   - Full system backup
   - Selective backup
   - Incremental backup
   - Compression

3. Cache:
   - Memory cache
   - Disk cache
   - Distributed cache
   - Cache policies

## Security Considerations

1. Authentication:
   - Token-based authentication
   - Role-based access control
   - Permission validation

2. Data Protection:
   - Encryption at rest
   - Encryption in transit
   - Secure backup storage

3. Rate Limiting:
   - Per-endpoint limits
   - User-based limits
   - Burst handling

## Monitoring

All endpoints are monitored for:
1. Response time
2. Error rates
3. Resource usage
4. Cache performance

Metrics are available through the monitoring API.