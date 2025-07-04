# Builtin Tools and Models

Agent-NN ships with a small set of builtin tools. They can be managed via the `agentnn tools` commands.

## Listing Tools

```bash
agentnn tools list
```

The output shows plugin based tools and builtin model wrappers.

## Inspecting a Tool

```bash
agentnn tools inspect agent_nn_v2
```

This prints the Python class used for the tool. Builtin models such as `agent_nn_v2` and the multi task reasoner can be referenced from agent configurations.
