"""Deploy Flowise nodes to a local installation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import httpx

from packaging.version import parse as parse_version

from tools.generate_flowise_plugin import generate_manifest
from tools.package_plugin import create_archive

DEFAULT_DEST = Path.home() / ".flowise" / "nodes" / "agent-nn"
VERSION_FILE = Path("plugin_version.txt")


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


def read_manifest_version(manifest: Path) -> str:
    return json.loads(manifest.read_text())[
        "version"
    ]


def should_deploy(new_version: str, dest: Path) -> bool:
    current = dest / VERSION_FILE.name
    if current.is_file():
        installed = current.read_text().strip()
        if parse_version(new_version) <= parse_version(installed):
            print(
                f"Installed plugin {installed} is newer or equal to {new_version}. Skipping."
            )
            return False
    return True


def deploy(src: Path, dest: Path, manifest: Path, reload_url: str | None) -> None:
    version = read_manifest_version(manifest)
    if not should_deploy(version, dest):
        return
    copy_files(src, dest)
    (dest / VERSION_FILE.name).write_text(version)
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

    manifest = generate_manifest(args.src, Path("flowise-plugin.json"))
    deploy(args.src, args.dest, manifest, args.reload_url)

    if args.build_plugin:
        create_archive(args.src, manifest, args.build_plugin)


if __name__ == "__main__":
    main()
