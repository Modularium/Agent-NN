# Plugin Agent API

The Plugin Agent executes registered tool plugins.

## `POST /execute_tool`

Execute a tool with given input.

**Body**
```json
{
  "tool_name": "filesystem",
  "input": {"action": "write", "path": "note.txt", "content": "hi"},
  "context": {}
}
```

Response contains the tool result.

## `GET /tools`

Return the list of available tool names.
