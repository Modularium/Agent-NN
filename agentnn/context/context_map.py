"""Generate a shared context map across sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..storage import context_store

__all__ = ["generate_map", "export_json", "export_html"]


def generate_map() -> Dict[str, List[Dict[str, Any]]]:
    """Return a graph representation of stored contexts."""
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    for sid in context_store.list_contexts():
        nodes.append({"id": sid, "type": "session"})
        for idx, entry in enumerate(context_store.load_context(sid)):
            node_id = f"{sid}_{idx}"
            nodes.append({"id": node_id, "type": "context"})
            edges.append({"source": sid, "target": node_id})
            agent = entry.get("agent_selection")
            if agent:
                nodes.append({"id": agent, "type": "agent"})
                edges.append({"source": node_id, "target": agent})
    return {"nodes": nodes, "edges": edges}


def export_json(path: str | Path) -> None:
    """Write the current context map as JSON."""
    data = generate_map()
    Path(path).write_text(json.dumps(data, indent=2))


_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>Context Map</title>
<script src='https://cdn.jsdelivr.net/npm/d3@7'></script></head>
<body><svg width='800' height='600'></svg>
<script>
const data = DATA_PLACEHOLDER;
const svg = d3.select('svg');
const links = data.edges.map(d => Object.create(d));
const nodes = data.nodes.map(d => Object.create(d));
const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(50))
    .force('charge', d3.forceManyBody().strength(-200))
    .force('center', d3.forceCenter(400,300));
const link = svg.append('g').selectAll('line')
    .data(links).enter().append('line')
    .attr('stroke', '#999');
const node = svg.append('g').selectAll('circle')
    .data(nodes).enter().append('circle')
    .attr('r', 5).attr('fill', '#69b');
simulation.on('tick', () => {
  link.attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);
  node.attr('cx', d => d.x).attr('cy', d => d.y);
});
</script></body></html>"""


def export_html(path: str | Path) -> None:
    """Write the current context map as an interactive HTML file."""
    data = generate_map()
    html = _HTML_TEMPLATE.replace("DATA_PLACEHOLDER", json.dumps(data))
    Path(path).write_text(html)
