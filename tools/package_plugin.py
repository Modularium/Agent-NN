"""Create an archive with Flowise nodes and manifest."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

DEFAULT_NODE_DIR = Path("integrations/flowise-nodes")
DEFAULT_MANIFEST = Path("flowise-plugin.json")
DEFAULT_OUTPUT = Path("agentnn_flowise_plugin.zip")


def create_archive(node_dir: Path, manifest: Path, output: Path) -> Path:
    """Bundle files into *output* archive and return the path."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for item in node_dir.glob("*.node.json"):
            shutil.copy2(item, tmp_dir / item.name)
        dist = node_dir / "dist"
        if dist.is_dir():
            for js in dist.glob("*.js"):
                shutil.copy2(js, tmp_dir / js.name)
        shutil.copy2(manifest, tmp_dir / manifest.name)
        base_name = output.with_suffix("")
        shutil.make_archive(str(base_name), output.suffix.lstrip("."), tmp_dir)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Flowise plugin")
    parser.add_argument("--node-dir", type=Path, default=DEFAULT_NODE_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    archive = create_archive(args.node_dir, args.manifest, args.output)
    print(f"Created {archive}")


if __name__ == "__main__":
    main()
