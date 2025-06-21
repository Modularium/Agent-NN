# API Einstieg

Die REST-Schnittstellen der Agent-NN-Services sind in OpenAPI beschrieben. Die wichtigsten Endpunkte sind unter `openapi/` dokumentiert.

- Registry: `/agents`, `/register`
- Dispatcher: `/task`
- Session Manager: `/start_session`, `/update_context`, `/context/{id}`
- Vector Store: `/add_document`, `/vector_search`
- LLM Gateway: `/generate`, `/embed`

Einen Gesamtüberblick liefert [openapi-overview.md](openapi-overview.md).
