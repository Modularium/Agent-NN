"""Execute a plugin from the command line."""

from __future__ import annotations

import argparse
import json

from plugins import PluginManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tool plugin")
    parser.add_argument("tool")
    parser.add_argument("--input", required=True, help="JSON input for the tool")
    args = parser.parse_args()

    data = json.loads(args.input)
    manager = PluginManager()
    plugin = manager.get(args.tool)
    if not plugin:
        print("Unknown tool")
        return
    result = plugin.execute(data, {})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
