# n8n AgentNN Node

This directory contains a custom n8n node that forwards tasks to the Agent-NN API gateway.
Run `npm install` followed by `npx tsc` to compile `AgentNN.node.ts` to JavaScript.
Copy the resulting files from `dist/` to your `~/.n8n/custom` directory and restart n8n.
Configure the `endpoint`, `taskType`, `method`, headers, timeout and `payload` parameters in the node settings.