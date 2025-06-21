# OpenAPI Overview

Version: v1.0.0-beta

| Method | Path | Description | Tags |
|-------|------|-------------|------|
| **GET** | `/agents` | List all registered agents | #agent #core |
| **POST** | `/register` | Register a new agent | #agent |
| **GET** | `/agents/{agent_id}` | Get agent by id | #agent |
| **POST** | `/task` | Queue a task for processing | #task |
| **POST** | `/start_session` | Start a new session | #session |
| **POST** | `/update_context` | Update conversation context | #session |
| **GET** | `/context/{session_id}` | Retrieve session context | #session |
| **POST** | `/add_document` | Add document to vector store | #vector |
| **POST** | `/vector_search` | Search documents via embeddings | #vector |
| **POST** | `/generate` | Generate text with selected model | #model |
| **POST** | `/embed` | Create vector embeddings | #vector |
