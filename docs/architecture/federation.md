# Federation Manager

The Federation Manager coordinates tasks across multiple Agent-NN clusters. Nodes
can be registered via the API and tasks are forwarded to the selected cluster.

```mermaid
graph TD
    APP[Local Dispatcher] --> FED[Federation Manager]
    FED --> NODE1[Remote Agent-NN]
    FED --> NODE2[Remote Agent-NN]
```

Use `/nodes` to register a remote instance and `/dispatch/{name}` to send tasks.
