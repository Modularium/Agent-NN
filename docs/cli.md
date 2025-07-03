# AgentNN CLI

The command line interface is now fully modular. Each logical group of features
is implemented as a subcommand under `sdk.cli.commands`.

Common examples:

```bash
python -m sdk.cli session start examples/three_agent_chain.yaml
python -m sdk.cli agent list
python -m sdk.cli model list
```

Session templates consist of an `agents` list and a sequence of `tasks`.
The tool prints results as JSON for easy scripting.
