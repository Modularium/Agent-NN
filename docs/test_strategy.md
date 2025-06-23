# Test Strategy

This project relies on pytest and pytest-cov for backend tests. Tests are grouped into
`tests/integration` for end-to-end flows, `tests/api` for REST routes and `tests/cli`
for the SDK commands. Optional React component tests live in `tests/ui` using vitest.

Coverage targets are 80% for all core services (`dispatcher`, `session`,
`llm_gateway`, `vector_store`, `routing_agent`). CLI commands and SDK helpers are
covered by unit tests with mocks. Regression tests document previously fixed bugs
such as the dispatcher legacy routing fallback.

Run `./tests/ci_check.sh` locally or via CI which executes ruff, mypy and pytest
with coverage output in HTML and JSON format.
