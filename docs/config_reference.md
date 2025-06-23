# Configuration Reference

This document lists the environment variables used by Agent-NN. Values can be defined in a `.env` file and are loaded via `pydantic-settings`.

| Variable | Description |
| --- | --- |
| OPENAI_API_KEY | API key for OpenAI models |
| DATABASE_URL | Connection string for the database |
| LLM_BACKEND | Backend used by the LLM Gateway |
| LLM_MODEL | Default language model |
| LLM_TEMPERATURE | Sampling temperature |
| LLM_MAX_TOKENS | Maximum tokens per request |
| VECTOR_STORE_URL | URL of the vector store service |
| EMBEDDING_MODEL | Model used for embeddings |
| LOG_LEVEL | Logging level |
| LOG_FORMAT | Logging format |
| LOG_JSON | Enable JSON logs |
| AUTOTRAINER_FREQUENCY_HOURS | Interval for the AutoTrainer |
| AUTH_ENABLED | Enable authentication in services |
| API_AUTH_ENABLED | Authentication for API gateway |
| RATE_LIMITS_ENABLED | Enable rate limiting |
| DATA_DIR | Base data directory |
| SESSIONS_DIR | Session storage location |
| VECTOR_DB_DIR | Vector database directory |
| MODELS_DIR | Directory for models |
| MLFLOW_TRACKING_URI | MLflow tracking server |

