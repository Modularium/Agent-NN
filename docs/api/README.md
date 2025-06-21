# API Overview

Die Agent-NN-Services stellen REST-Schnittstellen bereit. Die wichtigsten Endpunkte sind in den OpenAPI-Spezifikationen dokumentiert.

- Agent Registry: `/agents`, `/register`
- Task Dispatcher: `/task`
- Session Manager: `/start_session`, `/update_context`, `/context/{session_id}`
- Vector Store: `/add_document`, `/vector_search`
- LLM Gateway: `/generate`, `/embed`

Weitere Details finden sich in `openapi/` sowie im Dokument `openapi-overview.md`.
