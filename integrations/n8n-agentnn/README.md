# n8n AgentNN Node

This directory contains a custom n8n node which sends tasks to the Agent‑NN API
gateway. The node can be used to integrate Agent‑NN into your n8n workflows.

## Build

```bash
npm install
npm run build
```

The command creates the compiled files in the `dist/` directory. Copy these
files into your local `~/.n8n/custom` directory and restart n8n so that the node
is loaded.

## Usage

In the node parameters specify:

- **endpoint** – Base URL of the Agent‑NN API gateway.
- **taskType** – Type of task to execute (e.g. `chat`).
- **payload** – JSON payload passed as the `input` field.
- **path** – Optional API path appended to the endpoint (defaults to `/task`).
- **method** – HTTP method (`POST` or `GET`).
- **headers** – Optional additional HTTP headers as JSON object.
- **timeout** – Request timeout in milliseconds.

When executed, the node forwards the request to Agent‑NN and returns the API
response.
