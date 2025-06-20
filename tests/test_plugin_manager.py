from plugins import PluginManager


def test_load_plugins(tmp_path, monkeypatch):
    # create minimal plugin
    plugin_dir = tmp_path / "sample"
    plugin_dir.mkdir()
    (plugin_dir / "manifest.yaml").write_text("name: sample")
    (plugin_dir / "plugin.py").write_text(
        "from plugins import ToolPlugin\nclass Plugin(ToolPlugin):\n    def execute(self, input, context):\n        return {'ok': True}"
    )

    manager = PluginManager(str(tmp_path))
    assert "sample" in manager.list_plugins()
    plugin = manager.get("sample")
    assert plugin.execute({}, {}) == {"ok": True}
