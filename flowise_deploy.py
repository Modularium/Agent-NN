"""Deploy Flowise nodes to a local installation."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import httpx

from tools.generate_flowise_plugin import generate_manifest
from tools.package_plugin import create_archive

DEFAULT_DEST = Path.home() / ".flowise" / "nodes" / "agent-nn"


def copy_files(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.glob("*.node.json"):
        shutil.copy2(item, dest / item.name)
    dist = src / "dist"
    if dist.is_dir():
        for js in dist.glob("*.js"):
            shutil.copy2(js, dest / js.name)


def reload_nodes(url: str) -> None:
    try:
        httpx.post(url, timeout=5)
        print(f"Reload triggered via {url}")
    except Exception as exc:  # pragma: no cover - network errors
        print(f"Reload failed: {exc}")


def deploy(src: Path, dest: Path, reload_url: str | None) -> None:
    copy_files(src, dest)
    if reload_url:
        reload_nodes(reload_url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy Flowise nodes")
    parser.add_argument("--src", type=Path, default=Path("integrations/flowise-nodes"))
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--reload-url", help="Flowise reload endpoint")
    parser.add_argument(
        "--build-plugin",
        type=Path,
        metavar="ARCHIVE",
        help="Create a plugin archive after deployment",
    )
    args = parser.parse_args()

    deploy(args.src, args.dest, args.reload_url)

    if args.build_plugin:
        manifest = generate_manifest(args.src, Path("flowise-plugin.json"))
        create_archive(args.src, manifest, args.build_plugin)


if __name__ == "__main__":
    main()
