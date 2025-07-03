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

## Markers

All tests are tagged with either `@pytest.mark.unit` or `@pytest.mark.integration`. The default collection
runs only unit tests unless `--run-integration` is provided. This behavior is implemented in `tests/conftest.py`.

## Agent registry coverage

Iteration 3 adds focused unit tests for the `AgentRegistryService` and its API routes. The tests verify
metric counters, status handling and persistence of profile updates via the REST interface.
Using `tmp_path` ensures profile files are isolated.
An extra case validates that requesting an unknown agent increases the metrics counter and that the API responds with `404`.

## Supervisor agent coverage

Iteration 3.2 expands the unit tests for the `SupervisorAgent`. New cases
simulate a failing worker, verify the returned status for agents without any
history and assert that `_update_model` receives the correct success score for
both successful and failed executions. Dummy manager classes keep these tests
independent from the rest of the system.

## Fehlertests / Resilienz

Iteration 4 focuses on failure scenarios for the plugin agent service. Tests
cover unknown tool names and invalid plugin inputs so that the service returns
informative error messages instead of raising exceptions.
