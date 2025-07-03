# Session Runner CLI

All session commands are now available via the unified SDK CLI:

```
python -m sdk.cli session start examples/three_agent_chain.yaml
python -m sdk.cli session watch <session_id>
python -m sdk.cli session vote <session_id> --roles critic,reviewer
python -m sdk.cli session snapshot <session_id>
```

Sessions defined in a YAML template consist of an `agents` list and a sequence of
`tasks`. The tool prints results as JSON for easy scripting.
