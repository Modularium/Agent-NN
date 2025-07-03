from pathlib import Path

from agentnn.catalog.catalog_loader import load_catalog


def test_load_catalog(tmp_path: Path) -> None:
    sample = tmp_path / "a.yaml"
    sample.write_text("name: demo")
    items = load_catalog(tmp_path)
    assert items[0]["name"] == "demo"

