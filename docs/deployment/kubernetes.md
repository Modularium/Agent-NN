# Kubernetes Deployment

Helm chart is located at `deploy/k8s/helm/agent-nn`.

Install with:

```bash
helm install agent-nn deploy/k8s/helm/agent-nn -n agent-nn --create-namespace
```

This deploys the API gateway and backend services configured via `values.yaml`.
