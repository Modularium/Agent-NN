# Session Runner CLI

The `session_runner.py` tool simplifies starting and observing agent sessions.

```
python cli/session_runner.py start examples/three_agent_chain.yaml
python cli/session_runner.py watch <session_id>
python cli/session_runner.py vote <session_id> --roles critic,reviewer
python cli/session_runner.py snapshot <session_id>
```

Sessions defined in a YAML template consist of an `agents` list and a sequence of
`tasks`. The tool prints results as JSON for easy scripting.
