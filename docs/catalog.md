# Agent Catalog

Agent profiles can be stored as YAML files under `agents/catalog/`.
Each file contains metadata like `name`, `version`, `description`,
`license` and a list of `capabilities`.

Use `agentnn.catalog.catalog_loader.load_catalog()` to load all
entries from a directory.
