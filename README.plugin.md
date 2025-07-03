# Agent-NN Flowise Plugin

This plugin bundles all Flowise nodes for Agent-NN. It is versioned independently to allow safe updates via FlowiseHub.

## Build and package

```bash
python tools/generate_flowise_plugin.py
python tools/package_plugin.py --output agentnn_flowise_plugin.zip
```

The manifest includes `version` and `buildDate`. After packaging, deploy the plugin locally:

```bash
python flowise_deploy.py --dest ~/.flowise/nodes/agent-nn --build-plugin agentnn_flowise_plugin.zip
```

## Compatibility

| Plugin Version | Flowise >= |
| -------------- | ---------- |
| 1.0.4 | 1.3 |

The license follows the main repository (MIT).

See [docs/flowise_plugin.md](docs/flowise_plugin.md) for detailed instructions.
