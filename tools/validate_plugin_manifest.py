#!/usr/bin/env python
"""Validate flowise-plugin.json against a schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsonschema import Draft7Validator

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "version": {"type": "string"},
        "description": {"type": "string"},
        "author": {"type": "string"},
        "repository": {"type": "string"},
        "homepage": {"type": "string"},
        "license": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "nodes": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "required": [
        "name",
        "version",
        "description",
        "author",
        "repository",
        "homepage",
        "license",
        "keywords",
        "nodes",
    ],
}


def validate_manifest(path: Path) -> None:
    data = json.loads(path.read_text())
    validator = Draft7Validator(SCHEMA)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    if errors:
        for err in errors:
            loc = "/".join(str(p) for p in err.path) or "manifest"
            print(f"{loc}: {err.message}")
        raise SystemExit(1)
    print("Manifest valid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate flowise-plugin.json")
    parser.add_argument("manifest", type=Path, nargs="?", default=Path("flowise-plugin.json"))
    args = parser.parse_args()
    validate_manifest(args.manifest)


if __name__ == "__main__":
    main()

