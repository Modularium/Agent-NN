from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import List, Tuple


Command = Tuple[str, str, Path]


def _parse_module(
    path: Path,
) -> Tuple[dict[str, Tuple[str, str, Path]], List[Tuple[str, str, Path]]]:
    """Return Typer groups and root commands found in ``path``."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    groups: dict[str, Tuple[str, str, Path]] = {}
    root_cmds: List[Tuple[str, str, Path]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and func.attr == "Typer":
                kw = {
                    k.arg: getattr(k.value, "value", None) for k in node.value.keywords
                }
                name = kw.get("name")
                help_text = kw.get("help", "") or ""
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        groups[target.id] = (name or target.id, help_text, path)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            doc = ast.get_docstring(node) or ""
            for deco in node.decorator_list:
                if isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute):
                    if deco.func.attr == "command" and isinstance(
                        deco.func.value, ast.Name
                    ):
                        var = deco.func.value.id
                        cmd_name = None
                        if deco.args:
                            arg = deco.args[0]
                            if isinstance(arg, ast.Constant):
                                cmd_name = arg.value
                        for kw in deco.keywords:
                            if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                                cmd_name = kw.value.value
                        if cmd_name is None:
                            cmd_name = node.name
                        first_line = doc.splitlines()[0] if doc else ""
                        if var == "app" and path.name == "root.py":
                            root_cmds.append((cmd_name, first_line, path))
                        elif var in groups:
                            # command belongs to a group; ensure group stored
                            pass
    return groups, root_cmds


def collect_cli_info() -> Tuple[List[Command], List[Command]]:
    base = Path("sdk/cli/commands")
    groups: List[Command] = []
    root_cmds: List[Command] = []
    module_groups: dict[str, Tuple[str, str, Path]] = {}
    for path in sorted(base.glob("*.py")):
        g, r = _parse_module(path)
        module_groups.update(g)
        root_cmds.extend(r)

    main_tree = ast.parse(Path("sdk/cli/main.py").read_text())
    selected: List[str] = []
    import_map: dict[str, str] = {}
    register_map: dict[str, str] = {}
    for node in main_tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.endswith(tuple("commands".split()))
        ):
            mod = node.module.split(".")[-1]
            for alias in node.names:
                name = alias.asname or alias.name
                if alias.name == "register":
                    register_map[name] = mod
                else:
                    import_map[name] = mod
    for node in ast.walk(main_tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_typer"
        ):
            if node.args and isinstance(node.args[0], ast.Name):
                selected.append(node.args[0].id)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in register_map
        ):
            mod = register_map[node.func.id]
            reg_tree = ast.parse(Path(f"sdk/cli/commands/{mod}.py").read_text())
            for sub in ast.walk(reg_tree):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "add_typer"
                ):
                    if sub.args and isinstance(sub.args[0], ast.Name):
                        selected.append(sub.args[0].id)

    for var in selected:
        if var in module_groups:
            name, help_text, path = module_groups[var]
            groups.append((name, help_text, path))
    # deduplicate root commands
    root_unique: dict[str, Tuple[str, Path]] = {}
    for name, help_text, p in root_cmds:
        root_unique.setdefault(name, (help_text, p))
    root_cmds = [(n, h, p) for n, (h, p) in root_unique.items()]
    return groups, root_cmds


def generate_table(groups: List[Command], root_cmds: List[Command]) -> List[str]:
    lines = ["| Command | Description | Source |", "|---------|-------------|--------|"]
    docs_dir = Path("docs")
    for name, help_text, path in sorted(groups):
        rel = Path(os.path.relpath(path, docs_dir))
        lines.append(f"| `{name}` | {help_text} | [{path.name}]({rel.as_posix()}) |")
    for name, help_text, path in sorted(root_cmds):
        rel = Path(os.path.relpath(path, docs_dir))
        lines.append(f"| `{name}` | {help_text} | [{path.name}]({rel.as_posix()}) |")
    return lines


def update_cli_doc(output: Path) -> None:
    groups, root_cmds = collect_cli_info()
    table = generate_table(groups, root_cmds)
    text = output.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(text) if line.strip() == "## Subcommands")
    end = start + 1
    while end < len(text) and not text[end].startswith("## "):
        end += 1
    new_text = text[: start + 1] + ["", *table, ""] + text[end:]
    output.write_text("\n".join(new_text), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/cli.md")
    args = parser.parse_args()
    update_cli_doc(Path(args.output))
