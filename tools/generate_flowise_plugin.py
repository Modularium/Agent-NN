"""Generate a Flowise plugin manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_NODE_DIR = Path("integrations/flowise-nodes")
DEFAULT_OUTFILE = Path("flowise-plugin.json")


def collect_nodes(node_dir: Path) -> list[str]:
    """Return Flowise node names within *node_dir*."""
    nodes: list[str] = []
    for file in sorted(node_dir.glob("*.node.json")):
        data = json.loads(file.read_text())
        nodes.append(data.get("name", file.stem))
    return nodes


def generate_manifest(node_dir: Path, out_file: Path) -> Path:
    """Create plugin manifest from nodes in *node_dir* and store it at *out_file*."""
    version = Path("VERSION").read_text().strip()
    manifest = {
        "name": "agent-nn",
        "version": version,
        "description": "Agent-NN Flowise nodes",
        "nodes": collect_nodes(node_dir),
        "author": "Agent-NN Team",
        "keywords": ["agent-nn", "flowise", "plugin"],
    }
    out_file.write_text(json.dumps(manifest, indent=2))
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate flowise-plugin.json")
    parser.add_argument("--node-dir", type=Path, default=DEFAULT_NODE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTFILE)
    args = parser.parse_args()

    manifest_path = generate_manifest(args.node_dir, args.out)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
