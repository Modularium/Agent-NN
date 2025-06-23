# Metrics Reference

The following Prometheus metrics are exported by Agent-NN services.

- `agentnn_feedback_positive_total{agent}` – number of positive feedback entries per worker
- `agentnn_feedback_negative_total{agent}` – number of negative feedback entries per worker
- `agentnn_task_success_total{task_type}` – count of successful tasks per task type
- `agentnn_routing_decisions_total{task_type,worker}` – distribution of routing decisions

Use `/metrics` on each service to scrape these values.

## Test Coverage

All metrics-related utilities and services are covered by automated tests in
`tests/test_metrics.py` and related modules, ensuring stable behaviour.
