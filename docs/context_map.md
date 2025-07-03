# Shared Context Map

The context map visualises how sessions, agents and individual task results are connected.

The `context_map` module collects stored contexts from the configured storage backend and exports a graph structure in JSON.

```python
from agentnn.context import export_json, export_html

export_json("context_map.json")
export_html("context_map.html")
```

The JSON object contains `nodes` and `edges` and can be used to render a graph with D3.js or other libraries.
